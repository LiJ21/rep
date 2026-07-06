use arrow_array::{
    Array, Float64Array, Int64Array, RecordBatch, UInt32Array, UInt64Array, UInt8Array,
};
use orderbook_rs::{BookBatchGenerator, Row, Stats, UNDEF_PRICE};

const F_LAST: u8 = 128;
const F_TOB: u8 = 64;
const SNAPSHOT: u8 = 32;

fn row(action: u8, side: u8, price: i64, size: u32, order_id: u64, flags: u8) -> Row {
    Row {
        ts_event: 0,
        row_nr: 0,
        sequence: 0,
        publisher_id: 1,
        instrument_id: 7,
        channel_id: 0,
        action,
        side,
        price,
        size,
        order_id,
        flags,
    }
}

fn on(instrument_id: u32, mut r: Row) -> Row {
    r.instrument_id = instrument_id;
    r
}

#[derive(Debug)]
struct OutRow {
    row_nr: u64,
    instrument_id: u32,
    trade_px: Option<i64>,
    trade_sz: Option<u32>,
    trade_side: Option<u8>,
    trade_px_last: Option<i64>,
    trade_vwap: Option<f64>,
    trade_levels: Option<u32>,
    trade_fills: Option<u32>,
    trade_posted_sz: Option<u32>,
    bid_px: Vec<i64>,
    bid_sz: Vec<u64>,
    bid_ct: Vec<u32>,
    ask_px: Vec<i64>,
    ask_sz: Vec<u64>,
}

fn run(mut rows: Vec<Row>, levels: usize) -> (Vec<OutRow>, Stats) {
    for (i, r) in rows.iter_mut().enumerate() {
        r.row_nr = i as u64;
        r.sequence = i as u32;
        r.ts_event = 1_000 + i as i64;
    }
    let mut gen = BookBatchGenerator::new(rows.into_iter(), levels);
    let mut out = Vec::new();
    for batch in gen.by_ref() {
        parse(&batch.unwrap(), levels, &mut out);
    }
    (out, gen.stats().clone())
}

fn parse(batch: &RecordBatch, levels: usize, out: &mut Vec<OutRow>) {
    fn u32s<'a>(b: &'a RecordBatch, name: &str) -> &'a UInt32Array {
        col(b, name)
    }
    fn col<'a, T: 'static>(b: &'a RecordBatch, name: &str) -> &'a T {
        b.column(b.schema().index_of(name).unwrap())
            .as_any()
            .downcast_ref::<T>()
            .unwrap()
    }
    fn opt<A: Array, T>(a: &A, i: usize, get: impl Fn(&A, usize) -> T) -> Option<T> {
        (!a.is_null(i)).then(|| get(a, i))
    }

    for i in 0..batch.num_rows() {
        out.push(OutRow {
            row_nr: col::<UInt64Array>(batch, "row_nr").value(i),
            instrument_id: u32s(batch, "instrument_id").value(i),
            trade_px: opt(col::<Int64Array>(batch, "trade_px"), i, |a, i| a.value(i)),
            trade_sz: opt(u32s(batch, "trade_sz"), i, |a, i| a.value(i)),
            trade_side: opt(col::<UInt8Array>(batch, "trade_side"), i, |a, i| a.value(i)),
            trade_px_last: opt(col::<Int64Array>(batch, "trade_px_last"), i, |a, i| {
                a.value(i)
            }),
            trade_vwap: opt(col::<Float64Array>(batch, "trade_vwap"), i, |a, i| {
                a.value(i)
            }),
            trade_levels: opt(u32s(batch, "trade_levels"), i, |a, i| a.value(i)),
            trade_fills: opt(u32s(batch, "trade_fills"), i, |a, i| a.value(i)),
            trade_posted_sz: opt(u32s(batch, "trade_posted_sz"), i, |a, i| a.value(i)),
            bid_px: (0..levels)
                .map(|l| col::<Int64Array>(batch, &format!("bid_px_{l}")).value(i))
                .collect(),
            bid_sz: (0..levels)
                .map(|l| col::<UInt64Array>(batch, &format!("bid_sz_{l}")).value(i))
                .collect(),
            bid_ct: (0..levels)
                .map(|l| u32s(batch, &format!("bid_ct_{l}")).value(i))
                .collect(),
            ask_px: (0..levels)
                .map(|l| col::<Int64Array>(batch, &format!("ask_px_{l}")).value(i))
                .collect(),
            ask_sz: (0..levels)
                .map(|l| col::<UInt64Array>(batch, &format!("ask_sz_{l}")).value(i))
                .collect(),
        });
    }
}

#[test]
fn emits_only_when_top_n_changes() {
    let (out, stats) = run(
        vec![
            row(b'A', b'B', 100, 5, 1, F_LAST),
            row(b'A', b'B', 99, 3, 2, F_LAST),
            row(b'A', b'B', 98, 7, 3, F_LAST), // below top-2: no emit
            row(b'C', b'B', 98, 7, 3, F_LAST), // still below: no emit
            row(b'C', b'B', 100, 5, 1, F_LAST),
        ],
        2,
    );
    assert_eq!(stats.blocks, 5);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].bid_px, vec![100, UNDEF_PRICE]);
    assert_eq!(out[1].bid_px, vec![100, 99]);
    assert_eq!(out[2].bid_px, vec![99, UNDEF_PRICE]);
    assert_eq!(out[2].bid_sz, vec![3, 0]);
    assert!(out.iter().all(|r| r.trade_sz.is_none()));
}

#[test]
fn trade_block_full_fill() {
    let (out, stats) = run(
        vec![
            row(b'A', b'B', 100, 5, 1, F_LAST),
            row(b'A', b'A', 101, 4, 2, F_LAST),
            // sell aggressor 900 fully fills 2 lots of bid order 1
            row(b'T', b'A', 100, 2, 900, 0),
            row(b'F', b'B', 100, 2, 1, 0),
            row(b'C', b'B', 100, 2, 1, F_LAST),
        ],
        2,
    );
    assert_eq!(stats.trade_blocks, 1);
    assert_eq!(out.len(), 3);
    let t = &out[2];
    assert_eq!(t.trade_side, Some(1)); // sell aggressor
    assert_eq!(t.trade_sz, Some(2));
    assert_eq!(t.trade_px, Some(100));
    assert_eq!(t.trade_px_last, Some(100));
    assert_eq!(t.trade_vwap, Some(100.0));
    assert_eq!(t.trade_levels, Some(1));
    assert_eq!(t.trade_fills, Some(1));
    assert_eq!(t.trade_posted_sz, Some(0));
    assert_eq!(t.row_nr, 4); // block's last record
    assert_eq!(t.bid_px[0], 100);
    assert_eq!(t.bid_sz[0], 3); // post-trade book
    assert_eq!(t.ask_px[0], 101);
    assert_eq!(t.ask_sz[0], 4); // untouched ask side
}

#[test]
fn multi_level_sweep_aggregates() {
    let (out, stats) = run(
        vec![
            row(b'A', b'B', 100, 2, 1, F_LAST),
            row(b'A', b'B', 100, 1, 2, F_LAST),
            row(b'A', b'B', 99, 1, 3, F_LAST),
            // sell sweep through two levels, mirroring the real CME pattern
            row(b'T', b'A', 100, 3, 900, 0),
            row(b'F', b'B', 100, 2, 1, 0),
            row(b'F', b'B', 100, 1, 2, 0),
            row(b'T', b'A', 99, 1, 900, 0),
            row(b'F', b'B', 99, 1, 3, 0),
            row(b'C', b'B', 100, 2, 1, 0),
            row(b'C', b'B', 100, 1, 2, 0),
            row(b'C', b'B', 99, 1, 3, F_LAST),
        ],
        2,
    );
    assert_eq!(stats.trade_blocks, 1);
    let t = out.last().unwrap();
    assert_eq!(t.trade_sz, Some(4)); // sum of T sizes
    assert_eq!(t.trade_levels, Some(2));
    assert_eq!(t.trade_fills, Some(3));
    assert_eq!(t.trade_px, Some(100)); // sell: best = max
    assert_eq!(t.trade_px_last, Some(99)); // sell: deepest = min
    assert_eq!(t.trade_vwap, Some((100.0 * 3.0 + 99.0) / 4.0));
    assert_eq!(t.bid_px, vec![UNDEF_PRICE, UNDEF_PRICE]); // swept clean
}

#[test]
fn repriced_aggressor_self_fill_not_double_counted() {
    let (out, stats) = run(
        vec![
            row(b'A', b'A', 102, 1, 10, F_LAST),
            row(b'A', b'B', 100, 2, 20, F_LAST),
            // order 10 repriced into the bid: both sides get F records
            row(b'T', b'A', 100, 1, 10, 0),
            row(b'F', b'A', 100, 1, 10, 0), // aggressor's own fill: not counted
            row(b'F', b'B', 100, 1, 20, 0),
            row(b'M', b'B', 100, 1, 20, 0),
            row(b'C', b'A', 102, 1, 10, F_LAST),
        ],
        2,
    );
    assert_eq!(stats.trade_blocks, 1);
    let t = out.last().unwrap();
    assert_eq!(t.trade_sz, Some(1)); // from T only, not 2
    assert_eq!(t.trade_fills, Some(1)); // passive fill only
    assert_eq!(t.bid_px[0], 100);
    assert_eq!(t.bid_sz[0], 1);
    assert_eq!(t.ask_px[0], UNDEF_PRICE); // resting ask removed
}

#[test]
fn aggressor_remainder_posted() {
    let (out, _) = run(
        vec![
            row(b'A', b'A', 101, 1, 2, F_LAST),
            // buy aggressor 900 lifts the ask, remainder posts as a bid
            row(b'T', b'B', 101, 1, 900, 0),
            row(b'F', b'A', 101, 1, 2, 0),
            row(b'A', b'B', 100, 3, 900, 0),
            row(b'C', b'A', 101, 1, 2, F_LAST),
        ],
        2,
    );
    let t = out.last().unwrap();
    assert_eq!(t.trade_side, Some(0));
    assert_eq!(t.trade_posted_sz, Some(3));
    assert_eq!(t.bid_px[0], 100);
    assert_eq!(t.bid_sz[0], 3);
}

#[test]
fn snapshot_block_emits_once() {
    let (out, stats) = run(
        vec![
            row(b'R', b'N', UNDEF_PRICE, 0, 0, SNAPSHOT),
            row(b'A', b'B', 100, 1, 1, SNAPSHOT),
            row(b'A', b'A', 101, 1, 2, SNAPSHOT),
            row(b'A', b'B', 99, 1, 3, SNAPSHOT | F_LAST),
        ],
        2,
    );
    assert_eq!(stats.blocks, 1);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].bid_px, vec![100, 99]);
    assert_eq!(out[0].ask_px[0], 101);
}

#[test]
fn tob_add_replaces_side() {
    let (out, _) = run(
        vec![
            row(b'A', b'B', 100, 5, 0, F_TOB | F_LAST),
            row(b'A', b'B', UNDEF_PRICE, 0, 0, F_TOB | F_LAST), // clear side
            row(b'A', b'B', 101, 2, 0, F_TOB | F_LAST),
            row(b'A', b'B', 99, 4, 0, F_TOB | F_LAST), // replaces, not accumulates
        ],
        2,
    );
    assert_eq!(out.len(), 4);
    assert_eq!(out[0].bid_px[0], 100);
    assert_eq!(out[0].bid_sz[0], 5);
    assert_eq!(out[0].bid_ct[0], 1);
    assert_eq!(out[1].bid_px[0], UNDEF_PRICE);
    assert_eq!(out[2].bid_px[0], 101);
    assert_eq!(out[3].bid_px, vec![99, UNDEF_PRICE]);
    assert_eq!(out[3].bid_sz, vec![4, 0]);
}

#[test]
fn interleaved_instruments_finalize_independently() {
    let (out, stats) = run(
        vec![
            on(7, row(b'A', b'B', 100, 1, 1, 0)), // instrument 7, block open
            on(8, row(b'A', b'B', 200, 1, 2, F_LAST)), // instrument 8 finalizes
            on(7, row(b'A', b'A', 101, 1, 3, F_LAST)), // instrument 7 finalizes
        ],
        1,
    );
    assert_eq!(stats.blocks, 2);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].instrument_id, 8);
    assert_eq!(out[0].bid_px[0], 200);
    // instrument 7 emits once, with both its adds applied
    assert_eq!(out[1].instrument_id, 7);
    assert_eq!(out[1].bid_px[0], 100);
    assert_eq!(out[1].ask_px[0], 101);
}

#[test]
fn mixed_side_block_emits_one_row_per_side() {
    let (out, stats) = run(
        vec![
            row(b'A', b'B', 100, 1, 1, F_LAST),
            row(b'A', b'A', 101, 1, 2, F_LAST),
            // one packet, two aggressor sides
            row(b'T', b'A', 100, 1, 901, 0), // row_nr 2
            row(b'F', b'B', 100, 1, 1, 0),
            row(b'C', b'B', 100, 1, 1, 0),
            row(b'T', b'B', 101, 1, 902, 0),
            row(b'F', b'A', 101, 1, 2, 0),
            row(b'C', b'A', 101, 1, 2, F_LAST), // row_nr 7
        ],
        2,
    );
    assert_eq!(stats.mixed_side_trade_blocks, 1);
    let trades: Vec<_> = out.iter().filter(|r| r.trade_sz.is_some()).collect();
    assert_eq!(trades.len(), 2);
    // first-seen side first, metadata from its first T record
    assert_eq!(trades[0].trade_side, Some(1));
    assert_eq!(trades[0].row_nr, 2);
    // last side carries the block-final metadata
    assert_eq!(trades[1].trade_side, Some(0));
    assert_eq!(trades[1].row_nr, 7);
    // identical post-block book on both rows
    assert_eq!(trades[0].bid_px, trades[1].bid_px);
    assert_eq!(trades[0].ask_px, trades[1].ask_px);
    assert_eq!(trades[0].bid_px[0], UNDEF_PRICE);
    assert_eq!(trades[0].ask_px[0], UNDEF_PRICE);
}

#[test]
fn eof_without_f_last_flushes() {
    let (out, stats) = run(vec![row(b'A', b'B', 100, 1, 1, 0)], 1);
    assert_eq!(stats.unterminated_blocks, 1);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].bid_px[0], 100);
}

#[test]
fn zero_size_add_is_skipped() {
    let (out, stats) = run(
        vec![
            row(b'A', b'B', 100, 0, 1, F_LAST),
            row(b'A', b'B', 99, 1, 2, F_LAST),
        ],
        1,
    );
    assert_eq!(stats.zero_size_adds, 1);
    assert_eq!(out.len(), 1); // zero-size add emitted nothing
    assert_eq!(out[0].bid_px[0], 99);
}

#[test]
fn duplicate_add_and_missing_cancel_stats() {
    let (out, stats) = run(
        vec![
            row(b'A', b'B', 100, 5, 1, F_LAST),
            row(b'A', b'B', 101, 2, 1, F_LAST), // same id: replace
            row(b'C', b'B', 100, 1, 999, F_LAST), // unknown id
        ],
        1,
    );
    assert_eq!(stats.duplicate_adds, 1);
    assert_eq!(stats.missing_cancels, 1);
    assert_eq!(out.len(), 2);
    assert_eq!(out[1].bid_px[0], 101);
    assert_eq!(out[1].bid_sz[0], 2);
    assert_eq!(out[1].bid_ct[0], 1);
}

#[test]
fn trade_emits_even_when_top_unchanged() {
    // partial fill deep in the queue: size at top level shrinks, but if the
    // trade happens the row must still be emitted with trade columns
    let (out, _) = run(
        vec![
            row(b'A', b'B', 100, 5, 1, F_LAST),
            row(b'T', b'A', 100, 2, 900, 0),
            row(b'F', b'B', 100, 2, 1, 0),
            row(b'C', b'B', 100, 2, 1, F_LAST),
        ],
        1,
    );
    assert_eq!(out.len(), 2);
    assert_eq!(out[1].trade_sz, Some(2));
    assert_eq!(out[1].bid_sz[0], 3);
}
