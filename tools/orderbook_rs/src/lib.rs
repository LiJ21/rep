use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

const F_LAST: u8 = 128;
const F_TOB: u8 = 64;
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
    pub blocks: u64,
    pub trade_blocks: u64,
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
    pub mixed_side_trade_blocks: u64,
    pub sideless_trades: u64,
    pub zero_size_adds: u64,
    pub unterminated_blocks: u64,
    pub ts_regressions: u64,
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

#[derive(Clone, Copy)]
struct EmitMeta {
    ts_event: i64,
    row_nr: u64,
    sequence: u32,
}

/// Per-side trade aggregate over one F_LAST event block. Databento emits one
/// `T` per swept price level (order_id = aggressor) and one `F` per resting
/// order; `sum(T.size)` is the traded quantity while `F` can double-count
/// self-fills of a repriced aggressor.
struct TradeAgg {
    side: u8,
    qty: u64,
    px_qty: u64,
    notional: f64,
    px_min: i64,
    px_max: i64,
    px_first: i64,
    px_last: i64,
    levels: u32,
    fills: u32,
    posted: u64,
    ids: Vec<u64>,
    ts_event: i64,
    row_nr: u64,
    sequence: u32,
}

impl TradeAgg {
    fn new(row: &Row) -> Self {
        Self {
            side: row.side,
            qty: 0,
            px_qty: 0,
            notional: 0.0,
            px_min: i64::MAX,
            px_max: i64::MIN,
            px_first: UNDEF_PRICE,
            px_last: UNDEF_PRICE,
            levels: 0,
            fills: 0,
            posted: 0,
            ids: Vec::new(),
            ts_event: row.ts_event,
            row_nr: row.row_nr,
            sequence: row.sequence,
        }
    }

    fn record_trade(&mut self, row: &Row) {
        self.qty += row.size as u64;
        self.levels += 1;
        if row.price != UNDEF_PRICE {
            self.px_qty += row.size as u64;
            self.notional += row.price as f64 * row.size as f64;
            self.px_min = self.px_min.min(row.price);
            self.px_max = self.px_max.max(row.price);
            if self.px_first == UNDEF_PRICE {
                self.px_first = row.price;
            }
            self.px_last = row.price;
        }
        if row.order_id != 0 && !self.ids.contains(&row.order_id) {
            self.ids.push(row.order_id);
        }
    }

    fn px_best(&self) -> Option<i64> {
        (self.px_first != UNDEF_PRICE).then_some(match self.side {
            b'B' => self.px_min,
            b'A' => self.px_max,
            _ => self.px_first,
        })
    }

    fn px_worst(&self) -> Option<i64> {
        (self.px_first != UNDEF_PRICE).then_some(match self.side {
            b'B' => self.px_max,
            b'A' => self.px_min,
            _ => self.px_last,
        })
    }

    fn vwap(&self) -> Option<f64> {
        (self.px_qty > 0).then(|| self.notional / self.px_qty as f64)
    }
}

struct BookState {
    book: Book,
    last_top: Top,
    pending: Vec<TradeAgg>,
    dirty: bool,
    last_meta: EmitMeta,
}

impl BookState {
    fn new(levels: usize) -> Self {
        let book = Book::default();
        let last_top = book.top(levels);
        Self {
            book,
            last_top,
            pending: Vec::new(),
            dirty: false,
            last_meta: EmitMeta {
                ts_event: 0,
                row_nr: 0,
                sequence: 0,
            },
        }
    }
}

pub fn depth_schema(levels: usize) -> Arc<Schema> {
    Arc::new(schema(levels))
}

pub fn read_rows_parquet(
    path: &str,
) -> Result<impl Iterator<Item = Result<Row, ArrowError>>, Box<dyn std::error::Error>> {
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
    let reader = builder
        .with_projection(mask)
        .with_batch_size(262_144)
        .build()?;
    Ok(RowStream::new(reader))
}

pub fn read_rows_ipc<R: Read>(
    reader: R,
) -> Result<impl Iterator<Item = Result<Row, ArrowError>>, Box<dyn std::error::Error>> {
    Ok(RowStream::new(StreamReader::try_new(reader, None)?))
}

/// Streams rows batch-by-batch so a full day never has to be materialized.
pub struct RowStream<I> {
    batches: I,
    buf: std::vec::IntoIter<Row>,
    row_nr: u64,
    failed: bool,
}

impl<I: Iterator<Item = Result<RecordBatch, ArrowError>>> RowStream<I> {
    fn new(batches: I) -> Self {
        Self {
            batches,
            buf: Vec::new().into_iter(),
            row_nr: 0,
            failed: false,
        }
    }
}

impl<I: Iterator<Item = Result<RecordBatch, ArrowError>>> Iterator for RowStream<I> {
    type Item = Result<Row, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            if let Some(row) = self.buf.next() {
                return Some(Ok(row));
            }
            match self
                .batches
                .next()?
                .and_then(|batch| batch_rows(&batch, &mut self.row_nr))
            {
                Ok(rows) => self.buf = rows.into_iter(),
                Err(e) => {
                    self.failed = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

fn batch_rows(batch: &RecordBatch, row_nr: &mut u64) -> Result<Vec<Row>, ArrowError> {
    let c = InCols::new(batch)?;
    let mut rows = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        rows.push(Row {
            ts_event: timestamp_ns_at(c.ts_event, i)?,
            row_nr: *row_nr,
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
        *row_nr += 1;
    }
    Ok(rows)
}

pub fn write_depth_parquet<I>(
    rows: I,
    levels: usize,
    out: &str,
) -> Result<Stats, Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = Result<Row, ArrowError>>,
{
    let mut batches = BookBatchGenerator::new(rows.into_iter(), levels);
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(out)?, batches.schema(), Some(props))?;
    for batch in batches.by_ref() {
        writer.write(&batch?)?;
    }
    writer.close()?;
    Ok(batches.stats().clone())
}

pub fn write_depth_ipc<I, W: Write>(
    rows: I,
    levels: usize,
    out: W,
) -> Result<Stats, Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = Result<Row, ArrowError>>,
{
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
    books: HashMap<(u16, u32), BookState>,
    stats: Stats,
    prev_ts: i64,
    done: bool,
}

impl<I: Iterator<Item = Result<Row, ArrowError>>> BookBatchGenerator<I> {
    pub fn new(rows: I, levels: usize) -> Self {
        let schema = depth_schema(levels);
        Self {
            rows,
            levels,
            schema: schema.clone(),
            buf: OutBuffer::new(levels, schema),
            books: HashMap::new(),
            stats: Stats::default(),
            prev_ts: i64::MIN,
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
            match row {
                Ok(row) => self.push_row(row),
                Err(e) => {
                    self.done = true;
                    return Err(e);
                }
            }
            if self.buf.len() >= BATCH {
                return self.buf.take_batch();
            }
        }
        self.done = true;
        self.flush_unterminated();
        self.buf.take_batch()
    }

    fn push_row(&mut self, row: Row) {
        self.stats.rows += 1;
        if row.ts_event < self.prev_ts {
            self.stats.ts_regressions += 1;
        }
        self.prev_ts = row.ts_event;
        let key = (row.publisher_id, row.instrument_id);
        let levels = self.levels;
        let state = self
            .books
            .entry(key)
            .or_insert_with(|| BookState::new(levels));
        state.dirty = true;
        state.last_meta = EmitMeta {
            ts_event: row.ts_event,
            row_nr: row.row_nr,
            sequence: row.sequence,
        };
        match row.action {
            b'T' => {
                self.stats.trades_or_fills += 1;
                if row.side == b'N' {
                    self.stats.sideless_trades += 1;
                }
                let agg = match state.pending.iter_mut().position(|a| a.side == row.side) {
                    Some(i) => &mut state.pending[i],
                    None => {
                        state.pending.push(TradeAgg::new(&row));
                        state.pending.last_mut().unwrap()
                    }
                };
                agg.record_trade(&row);
            }
            b'F' => {
                self.stats.trades_or_fills += 1;
                if let Some(agg) = state
                    .pending
                    .iter_mut()
                    .find(|a| a.side != row.side && a.side != b'N')
                {
                    agg.fills += 1;
                }
            }
            b'N' => {
                self.stats.trades_or_fills += 1;
            }
            _ => {
                if row.action == b'A' && row.order_id != 0 {
                    for agg in state.pending.iter_mut() {
                        if agg.ids.contains(&row.order_id) {
                            agg.posted += row.size as u64;
                        }
                    }
                }
                apply(&mut state.book, row, &mut self.stats);
            }
        }
        if row.flags & F_LAST != 0 {
            finalize_block(state, key, levels, &mut self.buf, &mut self.stats);
        }
    }

    fn flush_unterminated(&mut self) {
        let mut keys: Vec<_> = self
            .books
            .iter()
            .filter(|(_, s)| s.dirty)
            .map(|(k, s)| (s.last_meta.row_nr, *k))
            .collect();
        keys.sort_unstable();
        for (_, key) in keys {
            self.stats.unterminated_blocks += 1;
            let state = self.books.get_mut(&key).unwrap();
            finalize_block(state, key, self.levels, &mut self.buf, &mut self.stats);
        }
    }
}

impl<I: Iterator<Item = Result<Row, ArrowError>>> Iterator for BookBatchGenerator<I> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch().transpose()
    }
}

/// Close one instrument's F_LAST event block: emit the post-block top once
/// per aggressor side present (trade columns per side, identical book
/// columns), or a single row when only the book changed. Duplicate rows for
/// earlier sides take their first T record's metadata so row_nr stays unique
/// and points at the trade record in the raw MBO.
fn finalize_block(
    state: &mut BookState,
    key: (u16, u32),
    levels: usize,
    buf: &mut OutBuffer,
    stats: &mut Stats,
) {
    stats.blocks += 1;
    let top = state.book.top(levels);
    let changed = top != state.last_top;
    if !state.pending.is_empty() {
        stats.trade_blocks += 1;
        if state.pending.len() > 1 {
            stats.mixed_side_trade_blocks += 1;
        }
        let n = state.pending.len();
        for (i, agg) in state.pending.iter().enumerate() {
            let meta = if i + 1 == n {
                state.last_meta
            } else {
                EmitMeta {
                    ts_event: agg.ts_event,
                    row_nr: agg.row_nr,
                    sequence: agg.sequence,
                }
            };
            buf.push(meta, key, &top, Some(agg));
            stats.emitted += 1;
        }
        state.pending.clear();
    } else if changed {
        buf.push(state.last_meta, key, &top, None);
        stats.emitted += 1;
    }
    if changed {
        state.last_top = top;
    }
    state.dirty = false;
}

fn apply(book: &mut Book, row: Row, stats: &mut Stats) {
    match row.action {
        b'R' => {
            stats.clears += 1;
            book.orders.clear();
            book.bids.clear();
            book.asks.clear();
        }
        b'A' => {
            stats.adds += 1;
            if !valid_side(row.side) {
                stats.bad_sides += 1;
                return;
            }
            if row.flags & F_TOB != 0 {
                let levels = side_levels_mut(book, row.side);
                levels.clear();
                if row.price != UNDEF_PRICE && row.size > 0 {
                    levels.insert(
                        row.price,
                        Level {
                            size: row.size as u64,
                            count: 1,
                        },
                    );
                }
                return;
            }
            if row.size == 0 {
                stats.zero_size_adds += 1;
                return;
            }
            if let Some(old) = book.orders.remove(&row.order_id) {
                stats.duplicate_adds += 1;
                remove_level_qty(book, old.side, old.price, old.size, true);
            }
            add_order(book, row.order_id, row.side, row.price, row.size);
        }
        b'C' => {
            stats.cancels += 1;
            let Some(mut order) = book.orders.get(&row.order_id).copied() else {
                stats.missing_cancels += 1;
                return;
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
        }
        b'M' => {
            stats.modifies += 1;
            if !valid_side(row.side) {
                stats.bad_sides += 1;
                return;
            }
            if let Some(old) = book.orders.remove(&row.order_id) {
                remove_level_qty(book, old.side, old.price, old.size, true);
            } else {
                stats.missing_modifies += 1;
            }
            add_order(book, row.order_id, row.side, row.price, row.size);
        }
        _ => {
            stats.unknown_actions += 1;
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
    if order_id == 0 || price == UNDEF_PRICE || size == 0 {
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
    trade_px_last: Vec<Option<i64>>,
    trade_vwap: Vec<Option<f64>>,
    trade_levels: Vec<Option<u32>>,
    trade_fills: Vec<Option<u32>>,
    trade_posted_sz: Vec<Option<u32>>,
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
            trade_px_last: Vec::with_capacity(BATCH),
            trade_vwap: Vec::with_capacity(BATCH),
            trade_levels: Vec::with_capacity(BATCH),
            trade_fills: Vec::with_capacity(BATCH),
            trade_posted_sz: Vec::with_capacity(BATCH),
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

    fn push(&mut self, meta: EmitMeta, key: (u16, u32), top: &Top, trade: Option<&TradeAgg>) {
        self.ts_event.push(meta.ts_event);
        self.row_nr.push(meta.row_nr);
        self.sequence.push(meta.sequence);
        self.publisher_id.push(key.0);
        self.instrument_id.push(key.1);
        self.trade_px.push(trade.and_then(|t| t.px_best()));
        self.trade_sz
            .push(trade.map(|t| t.qty.min(u32::MAX as u64) as u32));
        self.trade_side.push(trade.and_then(|t| side_code(t.side)));
        self.trade_px_last.push(trade.and_then(|t| t.px_worst()));
        self.trade_vwap.push(trade.and_then(|t| t.vwap()));
        self.trade_levels.push(trade.map(|t| t.levels));
        self.trade_fills.push(trade.map(|t| t.fills));
        self.trade_posted_sz
            .push(trade.map(|t| t.posted.min(u32::MAX as u64) as u32));
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
            Arc::new(Int64Array::from(std::mem::take(&mut self.trade_px_last))),
            Arc::new(Float64Array::from(std::mem::take(&mut self.trade_vwap))),
            Arc::new(UInt32Array::from(std::mem::take(&mut self.trade_levels))),
            Arc::new(UInt32Array::from(std::mem::take(&mut self.trade_fills))),
            Arc::new(UInt32Array::from(std::mem::take(
                &mut self.trade_posted_sz,
            ))),
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
        self.trade_px_last = Vec::with_capacity(BATCH);
        self.trade_vwap = Vec::with_capacity(BATCH);
        self.trade_levels = Vec::with_capacity(BATCH);
        self.trade_fills = Vec::with_capacity(BATCH);
        self.trade_posted_sz = Vec::with_capacity(BATCH);
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
        Field::new("trade_px_last", DataType::Int64, true),
        Field::new("trade_vwap", DataType::Float64, true),
        Field::new("trade_levels", DataType::UInt32, true),
        Field::new("trade_fills", DataType::UInt32, true),
        Field::new("trade_posted_sz", DataType::UInt32, true),
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
    fn new(batch: &'a RecordBatch) -> Result<Self, ArrowError> {
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

fn col<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T, ArrowError> {
    let i = batch.schema().index_of(name)?;
    batch.column(i).as_any().downcast_ref::<T>().ok_or_else(|| {
        ArrowError::SchemaError(format!(
            "column {name} has unexpected type {:?}",
            batch.schema().field(i).data_type()
        ))
    })
}

fn byte_at(array: &dyn Array, i: usize) -> Result<u8, ArrowError> {
    if array.is_null(i) {
        return Err(ArrowError::ParseError("null action/side".into()));
    }
    if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
        return a
            .value(i)
            .as_bytes()
            .first()
            .copied()
            .ok_or_else(|| ArrowError::ParseError("empty action/side".into()));
    }
    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
        return a
            .value(i)
            .as_bytes()
            .first()
            .copied()
            .ok_or_else(|| ArrowError::ParseError("empty action/side".into()));
    }
    Err(ArrowError::SchemaError(format!(
        "expected string action/side, got {:?}",
        array.data_type()
    )))
}

fn timestamp_ns_at(array: &dyn Array, i: usize) -> Result<i64, ArrowError> {
    if array.is_null(i) {
        return Err(ArrowError::ParseError("null ts_event".into()));
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
    Err(ArrowError::SchemaError(format!(
        "expected timestamp ts_event, got {:?}",
        array.data_type()
    )))
}

fn side_code(side: u8) -> Option<u8> {
    match side {
        b'B' => Some(0),
        b'A' => Some(1),
        _ => None,
    }
}
