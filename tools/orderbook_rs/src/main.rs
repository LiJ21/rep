use std::env;
use std::io;

use orderbook_rs::{
    read_rows_ipc, read_rows_parquet, sort_rows, write_depth_ipc, write_depth_parquet, Stats,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let levels = value(&args, "--levels")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let format = value(&args, "--format").unwrap_or_else(|| "parquet".to_string());
    let mut rows = if has(&args, "--input-ipc") {
        read_rows_ipc(io::stdin().lock())?
    } else {
        read_rows_parquet(input_path(&args).ok_or("missing input parquet path")?)?
    };
    sort_rows(&mut rows);

    let stats = match format.as_str() {
        "ipc" | "arrow" => write_depth_ipc(rows, levels, io::stdout().lock())?,
        "parquet" => {
            let out = value(&args, "--out").unwrap_or_else(|| "book_depth.parquet".to_string());
            write_depth_parquet(rows, levels, &out)?
        }
        _ => return Err(format!("unsupported --format {format:?}; use parquet or ipc").into()),
    };
    print_stats(&stats);
    Ok(())
}

fn print_stats(s: &Stats) {
    eprintln!(
        "rows={} emitted={} add={} cancel={} modify={} clear={} trade_or_fill={}",
        s.rows, s.emitted, s.adds, s.cancels, s.modifies, s.clears, s.trades_or_fills
    );
    eprintln!(
        "warnings missing_cancels={} missing_modifies={} duplicate_adds={} bad_sides={} unknown_actions={}",
        s.missing_cancels, s.missing_modifies, s.duplicate_adds, s.bad_sides, s.unknown_actions
    );
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn value(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn input_path(args: &[String]) -> Option<&str> {
    let mut i = 1;
    while i < args.len() {
        if matches!(args[i].as_str(), "--levels" | "--out" | "--format") {
            i += 2;
        } else if args[i].starts_with("--") {
            i += 1;
        } else {
            return Some(&args[i]);
        }
    }
    None
}
