import warnings
warnings.filterwarnings('ignore')

import plotly
import plotly.express as px
import plotly.utils
from flask import Flask, render_template_string, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import random
import re
from glob import glob
import os
from datetime import datetime, timedelta
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import csv
import sqlite3

app = Flask(__name__)

# Cache configuration
# In-memory cache keeps the current dashboard fast.
# Incremental disk cache prevents re-reading unchanged CSV files from
# the Network Drive every time the in-memory cache expires or the app restarts.
CACHE_TTL = 300

# How many files to read from the Network Drive at once. UNC/SMB reads are
# I/O-bound (waiting on the network, not the CPU), so a thread pool gives a
# large speedup even though Python has a GIL. Tune down if the network share
# starts throttling/timing out with many concurrent connections.
NETWORK_READ_WORKERS = 16

# When no specific single date is selected, only load files from the last
# N days by default (see get_files_from_line). Some line folders have
# 50,000+ historical files, and loading all of them every request is the
# main cause of slow first-loads. Change this if 3 days is too little/much.
DEFAULT_RECENT_DAYS = 30
data_cache = {}
cache_lock = threading.Lock()

# ------------------------------------------------------------
# INCREMENTAL DISK CACHE
# ------------------------------------------------------------
CACHE_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "smt_cache.db"
)
CACHE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "smt_cache_data"
)
os.makedirs(CACHE_DATA_DIR, exist_ok=True)

def init_cache_db():
    """Create the metadata database used by the incremental CSV cache."""
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS imported_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_name TEXT NOT NULL,
                log_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL DEFAULT 0,
                modified_time REAL NOT NULL DEFAULT 0,
                cache_file TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                imported_time TEXT,
                UNIQUE(line_name, log_type, file_path)
            )
        """)
        conn.commit()
    finally:
        conn.close()

init_cache_db()

def get_disk_cache_path(line_name, log_type, file_path):
    """Return a safe local pickle path for one source CSV file."""
    raw = f"{line_name}|{log_type}|{os.path.normcase(os.path.abspath(file_path))}"
    digest = hashlib.sha256(raw.encode('utf-8', errors='ignore')).hexdigest()
    return os.path.join(CACHE_DATA_DIR, f"{digest}.pkl")

def get_file_info(file_path):
    try:
        stat = os.stat(file_path)
        return stat.st_size, stat.st_mtime
    except (OSError, IOError):
        return None

def get_cached_file_record(line_name, log_type, file_path):
    conn = sqlite3.connect(CACHE_DB)
    try:
        cur = conn.execute("""
            SELECT file_size, modified_time, cache_file, row_count
            FROM imported_files
            WHERE line_name = ? AND log_type = ? AND file_path = ?
        """, (line_name, log_type, file_path))
        return cur.fetchone()
    finally:
        conn.close()

def get_all_cached_file_records(line_name, log_type):
    """
    Load every cache record for this line/log_type in a single query,
    instead of opening a new sqlite connection per file. Returns a dict
    keyed by file_path: {file_path: (file_size, modified_time, cache_file, row_count)}
    """
    conn = sqlite3.connect(CACHE_DB)
    try:
        cur = conn.execute("""
            SELECT file_path, file_size, modified_time, cache_file, row_count
            FROM imported_files
            WHERE line_name = ? AND log_type = ?
        """, (line_name, log_type))
        return {row[0]: row[1:] for row in cur.fetchall()}
    finally:
        conn.close()

def save_cached_file_records_batch(records):
    """
    Batch upsert of cache records in a single connection/transaction.
    records: list of tuples matching the INSERT column order below.
    """
    if not records:
        return
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.executemany("""
            INSERT INTO imported_files (
                line_name, log_type, file_path, file_name,
                file_size, modified_time, cache_file, row_count, imported_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(line_name, log_type, file_path) DO UPDATE SET
                file_name = excluded.file_name,
                file_size = excluded.file_size,
                modified_time = excluded.modified_time,
                cache_file = excluded.cache_file,
                row_count = excluded.row_count,
                imported_time = excluded.imported_time
        """, records)
        conn.commit()
    finally:
        conn.close()

def save_cached_file_record(line_name, log_type, file_path, cache_file,
                            file_size, modified_time, row_count):
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.execute("""
            INSERT INTO imported_files (
                line_name, log_type, file_path, file_name,
                file_size, modified_time, cache_file, row_count, imported_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(line_name, log_type, file_path) DO UPDATE SET
                file_name = excluded.file_name,
                file_size = excluded.file_size,
                modified_time = excluded.modified_time,
                cache_file = excluded.cache_file,
                row_count = excluded.row_count,
                imported_time = excluded.imported_time
        """, (
            line_name, log_type, file_path, os.path.basename(file_path),
            file_size, modified_time, cache_file, row_count,
            datetime.now().isoformat(timespec='seconds')
        ))
        conn.commit()
    finally:
        conn.close()

def load_cached_dataframe(cache_file):
    try:
        if not os.path.exists(cache_file):
            return None
        df = pd.read_pickle(cache_file)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception as e:
        print(f"Cache read error: {cache_file} -> {e}")
    return None

def save_cached_dataframe(df, cache_file):
    try:
        tmp_file = cache_file + '.tmp'
        df.to_pickle(tmp_file, protocol=4)
        os.replace(tmp_file, cache_file)
        return True
    except Exception as e:
        print(f"Cache write error: {cache_file} -> {e}")
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        return False

DETAIL_LIMIT = None

LINE_CONFIG = {
    "Line 1": {
        "paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line1-1\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line1-2\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line1-3\RetryLog"
        ],
        "file_pattern": "*.csv",
        "year": 2026,
        "cycle_time_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line1-1\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line1-2\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line1-3\ProgramLog"
        ],
        "cycle_time_pattern": "LotLog*.csv",
        "solder_paste_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line1_0"
        ],
        # OP018LotLog*.csv is the pattern originally assumed for this line;
        # LotLog*.csv added as a fallback since Line 3 turned out to use
        # that naming instead (confirmed 18/8/2026). Whichever pattern
        # actually matches files on disk is the one that gets used —
        # having both costs nothing if one glob simply returns no files.
        "solder_paste_pattern": ["OP018LotLog*.csv", "LotLog*.csv"]
    },
    "Line 2": {
        "paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line2-1\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line2-2\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line2-3\RetryLog"
        ],
        "file_pattern": "*.csv",
        "year": 2026,
        "cycle_time_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line2-1\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line2-2\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line2-3\ProgramLog"
        ],
        "cycle_time_pattern": "LotLog*.csv",
        "solder_paste_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line2_0"
        ],
        "solder_paste_pattern": ["OP018LotLog*.csv", "LotLog*.csv"]
    },
    "Line 3": {
        "paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line3-1\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line3-2\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line3-3\RetryLog"
        ],
        "file_pattern": "*.csv",
        "year": 2026,
        "cycle_time_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line3-1\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line3-2\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line3-3\ProgramLog"
        ],
        "cycle_time_pattern": "LotLog*.csv",
        "solder_paste_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line3_0"
        ],
        # Confirmed 18/8/2026: Line 3's printer actually exports as
        # LotLog*.csv (same naming as its cycle-time ProgramLog files),
        # NOT OP018LotLog like the other lines, and NOT PcbLog as
        # previously assumed. Header layout/columns are identical to
        # OP018LotLog (header=1, same Program Name/Setup Date/Finish
        # Date/Print CT AVE/Cleaning Time/Cleaning Count column names),
        # so detect_columns_solder_paste and read_csv_file needed no
        # changes — only this glob pattern was wrong. LotLog*.csv listed
        # first since it's the confirmed real filename; the others kept
        # as fallbacks in case firmware changes again.
        "solder_paste_pattern": ["LotLog*.csv", "OP018LotLog*.csv", "PcbLog*.csv"]
    },
    "Line 4": {
        "paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line4-1\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line4-2\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line4-3\RetryLog"
        ],
        "file_pattern": "*.csv",
        "year": 2026,
        "cycle_time_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line4-1\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line4-2\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line4-3\ProgramLog"
        ],
        "cycle_time_pattern": "LotLog*.csv",
        "solder_paste_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line4_0"
        ],
        "solder_paste_pattern": ["OP018LotLog*.csv", "LotLog*.csv"]
    },
    "Line 5": {
        "paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line5-1\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line5-2\RetryLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line5-3\RetryLog"
        ],
        "file_pattern": "*.csv",
        "year": 2026,
        "cycle_time_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line5-1\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line5-2\ProgramLog",
            r"\\10.52.61.102\TraceabilityLogTemp\Line5-3\ProgramLog"
        ],
        "cycle_time_pattern": "LotLog*.csv",
        "solder_paste_paths": [
            r"\\10.52.61.102\TraceabilityLogTemp\Line5_0"
        ],
        "solder_paste_pattern": ["OP018LotLog*.csv", "LotLog*.csv"]
    }
}

def extract_machine_no(file_path):
    if not file_path:
        return 'UNKNOWN'
    match = re.search(r'(Line\d+[-_]?\d+)', file_path, re.IGNORECASE)
    if match:
        return match.group(1)
    parts = re.split(r'[\\/]', file_path)
    for p in reversed(parts):
        if re.match(r'^Line\d+[-_]?\d+$', p, re.IGNORECASE):
            return p
    return 'UNKNOWN'

def get_cache_key(line_name, log_type, filters=None):
    key = f"{line_name}_{log_type}"
    if filters:
        key += f"_{json.dumps(filters, sort_keys=True)}"
    return hashlib.md5(key.encode()).hexdigest()

def normalize_path(path):
    if path.startswith('//'):
        path = path.replace('/', '\\')
    if not path.endswith('\\') and not path.endswith('/'):
        path = path + '\\'
    return path

def extract_date_from_filename(basename):
    """
    Pull an embedded YYYYMMDD date out of a log filename, regardless of
    naming convention (OP018LotLog20260817142933.csv, LotLog20260817.csv,
    ErrLog20260818.csv, PrinterSetupLogY56219_20260818.csv, ...).
    Returns a date object, or None if no 8-digit date could be found/parsed.
    """
    match = re.search(r'(\d{8})', basename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), '%Y%m%d').date()
    except ValueError:
        return None

def get_files_from_line(line_name, log_type="Retry Log", date_filter=None, recent_days=DEFAULT_RECENT_DAYS):
    line_data = LINE_CONFIG[line_name]

    if log_type == "Cycle Time Log":
        paths = line_data.get('cycle_time_paths', [])
        if date_filter:
            patterns = [f"LotLog{date_filter}*.csv"]
        else:
            cycle_pattern = line_data.get('cycle_time_pattern', 'LotLog*.csv')
            patterns = cycle_pattern if isinstance(cycle_pattern, list) else [cycle_pattern]
        print(f"[Cycle Time] Reading from: {paths}")
    elif log_type == "Solder Paste Log":
        paths = line_data.get('solder_paste_paths', [])
        if date_filter:
            base_pattern = line_data.get('solder_paste_pattern', 'OP018LotLog*.csv')
            base_patterns = base_pattern if isinstance(base_pattern, list) else [base_pattern]
            # Swap the trailing "*.csv" for "{date_filter}*.csv" on each
            # candidate pattern's prefix (text before the first '*').
            patterns = []
            for p in base_patterns:
                prefix = p.split('*')[0]
                patterns.append(f"{prefix}{date_filter}*.csv")
        else:
            base_pattern = line_data.get('solder_paste_pattern', 'OP018LotLog*.csv')
            patterns = base_pattern if isinstance(base_pattern, list) else [base_pattern]
        print(f"[Solder Paste] Reading from: {paths} (patterns: {patterns})")
    else:
        paths = line_data['paths']
        if log_type == "Error Log":
            paths = [p.replace("RetryLog", "ErrorLog") for p in paths]
        patterns = [line_data.get('file_pattern', '*.csv')]
        print(f"[{log_type}] Reading from: {paths}")

    all_files = []
    for path in paths:
        path = normalize_path(path)
        for pattern in patterns:
            try:
                files = glob(os.path.join(path, pattern))
                all_files.extend(files)
            except Exception as e:
                print(f"Error reading {path} (pattern {pattern}): {e}")
                continue

    print(f"Found {len(all_files)} files")

    # When the caller didn't ask for one specific date, a folder can easily
    # contain tens of thousands of historical files (seen: 60,775 on one
    # line). Loading everything on every request is what made requests slow
    # even with per-file caching. Default to only the most recent N days,
    # using the date embedded in the filename itself (no network/stat call
    # needed to filter). Pass recent_days=None to disable and load everything.
    if not date_filter and recent_days is not None:
        cutoff = datetime.now().date() - timedelta(days=recent_days - 1)
        recent_files = []
        undated_files = []
        for fp in all_files:
            fdate = extract_date_from_filename(os.path.basename(fp))
            if fdate is None:
                undated_files.append(fp)
            elif fdate >= cutoff:
                recent_files.append(fp)
        # Keep undated files too (better to include a few extra than to
        # silently drop data we couldn't classify), but don't let them
        # defeat the point of filtering if there happen to be many.
        all_files = recent_files + undated_files
        print(f"Filtered to last {recent_days} day(s) (since {cutoff}): {len(recent_files)} dated + {len(undated_files)} undated = {len(all_files)} files")

    return all_files

def get_available_dates(line_name):
    line_data = LINE_CONFIG[line_name]
    paths = line_data.get('cycle_time_paths', [])
    dates = []

    for path in paths:
        path = normalize_path(path)
        try:
            files = glob(os.path.join(path, "LotLog*.csv"))
            for f in files:
                basename = os.path.basename(f)
                if basename.startswith('LotLog') and basename.endswith('.csv'):
                    date_str = basename[6:14]
                    if len(date_str) == 8 and date_str.isdigit():
                        formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        dates.append({
                            'raw': date_str,
                            'display': formatted,
                            'file': basename
                        })
        except Exception:
            continue

    return sorted(dates, key=lambda x: x['raw'], reverse=True)

def get_available_solder_paste_dates(line_name):
    line_data = LINE_CONFIG[line_name]
    paths = line_data.get('solder_paste_paths', [])
    raw_pattern = line_data.get('solder_paste_pattern', 'OP018LotLog*.csv')
    patterns = raw_pattern if isinstance(raw_pattern, list) else [raw_pattern]
    dates = []
    seen_raw = set()

    for path in paths:
        path = normalize_path(path)
        for pattern in patterns:
            try:
                files = glob(os.path.join(path, pattern))
            except Exception:
                continue
            for f in files:
                basename = os.path.basename(f)
                # Generic: pull the 8-digit YYYYMMDD out of the filename
                # regardless of naming convention (OP018LotLog... vs
                # PcbLog... etc.) instead of assuming a fixed prefix length.
                fdate = extract_date_from_filename(basename)
                if fdate is None:
                    continue
                date_str = fdate.strftime('%Y%m%d')
                if date_str in seen_raw:
                    continue
                seen_raw.add(date_str)
                dates.append({
                    'raw': date_str,
                    'display': fdate.strftime('%Y-%m-%d'),
                    'file': basename
                })

    return sorted(dates, key=lambda x: x['raw'], reverse=True)

def generate_sample_data(line_name, num_records=1000, log_type="Retry Log"):
    if log_type == "Cycle Time Log":
        programs = ["PROG-A001", "PROG-B002", "PROG-C003", "PROG-D004", "PROG-E005"]
        stations = ["Printer", "SPI", "P&P-1", "P&P-2", "Reflow", "AOI", "Inspection"]
        lots = [f"LOT{random.randint(10000, 99999)}" for _ in range(10)]
        line_no = line_name.replace(' ', '')
        data = []
        for i in range(num_records):
            month = random.randint(1, 5)
            day = random.randint(1, 28)
            start_ts = datetime(2026, month, day, random.randint(0, 23), random.randint(0, 59))
            cycle_time = round(random.uniform(12, 45), 2)
            finish_ts = start_ts + timedelta(seconds=cycle_time)
            lot = random.choice(lots)
            machine_no = f"{line_no}-{random.randint(1, 3)}"
            data.append({
                'Start Date': start_ts.strftime('%Y-%m-%d %H:%M:%S'),
                'Finish Date': finish_ts.strftime('%Y-%m-%d %H:%M:%S'),
                'Production Lot': lot,
                'Program Name': random.choice(programs),
                'Station': random.choice(stations),
                'Cycle Time': cycle_time,
                'Status': random.choice(['PASS', 'PASS', 'PASS', 'FAIL']),
                'Operator': f"OP{random.randint(1, 20):03d}",
                'line': line_name,
                'Machine No': machine_no,
                'Path': f"[demo] \\\\10.52.61.102\\TraceabilityLogTemp\\{machine_no}\\ProgramLog\\LotLog{start_ts.strftime('%Y%m%d')}.csv",
            })
        return pd.DataFrame(data)
    elif log_type == "Solder Paste Log":
        programs = ["KOBAA03_TOP", "KOBAA03_BOT", "KOBAA05_TOP", "KOBAA05_BOT"]
        data = []
        line_no = line_name.replace(' ', '')
        for i in range(num_records):
            month = random.randint(1, 5)
            day = random.randint(1, 28)
            setup_ts = datetime(2026, month, day, random.randint(8, 10), random.randint(0, 59))
            finish_ts = setup_ts + timedelta(hours=random.randint(2, 8))
            data.append({
                'Program Name': random.choice(programs),
                'Setup Date': setup_ts.strftime('%d/%m/%Y %H:%M'),
                'Finish Date': finish_ts.strftime('%d/%m/%Y %H:%M'),
                'Print CT AVE': round(random.uniform(10, 15), 2),
                'Cleaning Time': round(random.uniform(0.5, 5), 2),
                'Cleaning Count': random.randint(0, 10),
                'Machine No': f"{line_no}_0",
                'Path': f"[demo] \\\\10.52.61.102\\TraceabilityLogTemp\\{line_no}_0\\OP018LotLog{setup_ts.strftime('%Y%m%d%H%M%S')}.csv",
            })
        return pd.DataFrame(data)
    else:
        error_types = [
            "Pick & Place Error", "Solder Paste Insufficient", "Component Missing",
            "Alignment Error", "Vision System Error", "Temperature Out of Range",
            "Conveyor Jam", "Nozzle Clog", "Vacuum Leak", "PCB Warpage",
            "Feeder Error", "Height Sensor Error", "Mark Recognition Error", "Skip Error"
        ]
        lots = [f"LOT{random.randint(10000, 99999)}" for _ in range(1000)]
        line_no = line_name.replace(' ', '')
        folder = "ErrorLogLog" if log_type == "Error Log" else "RetryLog"
        data = []
        for i in range(num_records):
            month = random.randint(1, 5)
            ts = datetime(2026, month, random.randint(1, 28), random.randint(0, 23), random.randint(0, 59))
            machine_no = f"{line_no}-{random.randint(1, 3)}"
            data.append({
                'Occurrence Time': ts,
                'Error Name': random.choice(error_types),
                'Lot No': random.choice(lots),
                'line': line_name,
                'Machine No': machine_no,
                'Path': f"[demo] \\\\10.52.61.102\\TraceabilityLogTemp\\{machine_no}\\{folder}\\{ts.strftime('%Y%m%d')}.csv",
            })
        return pd.DataFrame(data)

def read_csv_file(fp):
    print(f"Reading: {os.path.basename(fp)}")
    encodings = ['utf-8', 'latin1', 'cp874', 'windows-874', 'tis-620']
    # These YAMAHA log exports have a throwaway first line before the real
    # header row, so they need header=1. Confirmed for LotLog/OP018LotLog;
    # PcbLog (Line 3's printer log) is assumed to follow the same export
    # format until we see a real sample — if Line 3's Solder Paste Log comes
    # back empty/misaligned, this is the first thing to check.
    lotlog_prefixes = ('lotlog', 'op018lotlog', 'pcblog')
    is_lotlog = os.path.basename(fp).lower().startswith(lotlog_prefixes)

    for enc in encodings:
        try:
            if is_lotlog:
                df = pd.read_csv(fp, header=1, encoding=enc, nrows=50000, on_bad_lines='skip')
            else:
                df = pd.read_csv(fp, encoding=enc, nrows=50000, on_bad_lines='skip')

            if len(df) > 0 and len(df.columns) > 1:
                print(f"  Success: {os.path.basename(fp)} ({len(df)} rows, {len(df.columns)} cols)")
                return df
        except Exception as e:
            print(f"  Read error [{enc}]: {e}")

    print(f"  Failed: {os.path.basename(fp)}")
    return None

def _read_and_tag(fp, line_name):
    """
    Read one file from the Network Drive and tag it with source metadata.
    Designed to run inside a ThreadPoolExecutor worker: reading over UNC/SMB
    is network-bound, so many of these can run concurrently.
    Returns (file_path, dataframe_or_None).
    """
    d = read_csv_file(fp)
    if d is None or len(d) == 0:
        return fp, None
    d['source_file'] = os.path.basename(fp)
    d['Path'] = fp
    d['Machine No'] = extract_machine_no(fp)
    return fp, d

def load_data_from_line(line_name, log_type, date_filter=None, recent_days=DEFAULT_RECENT_DAYS):
    """
    Load data using two cache layers:
      1) in-memory cache for repeated dashboard requests
      2) per-file disk cache for unchanged CSV files

    Only new or modified CSV files are read from the Network Drive.
    """
    cache_key = get_cache_key(
        line_name,
        log_type,
        {'date': date_filter, 'recent_days': recent_days} if date_filter else {'recent_days': recent_days}
    )

    # Fast in-memory cache.
    with cache_lock:
        cached = data_cache.get(cache_key)
        if cached:
            cached_data, timestamp = cached
            if (datetime.now() - timestamp).total_seconds() < CACHE_TTL:
                print(f"Memory cache hit: {line_name} / {log_type}")
                return cached_data
            # Remove expired entry so the object does not stay around forever.
            data_cache.pop(cache_key, None)

    print(f"Loading {log_type} data from {line_name}")
    files = get_files_from_line(line_name, log_type, date_filter, recent_days=recent_days)

    if not files:
        print("No files found, generating sample data")
        df = generate_sample_data(line_name, 1000, log_type)
        result = (df, ["[demo data]"])
        with cache_lock:
            data_cache[cache_key] = (result, datetime.now())
        return result

    all_dfs = []
    files_loaded_from_cache = 0
    files_read_from_network = 0

    # ---- Pass 1: one sqlite query for ALL records of this line/log_type,
    # instead of opening a new connection per file (this was the main cost
    # when there are hundreds of thousands of small lot-log files). ----
    cache_records = get_all_cached_file_records(line_name, log_type)

    files_to_read = []
    file_infos = {}

    for fp in files:
        info = get_file_info(fp)
        if info is None:
            print(f"Cannot stat file, skipping: {fp}")
            continue
        file_infos[fp] = info
        file_size, modified_time = info

        record = cache_records.get(fp)
        if record:
            old_size, old_modified, old_cache_file, old_row_count = record
            if (
                old_cache_file
                and old_size == file_size
                and abs(old_modified - modified_time) < 0.01
            ):
                cached_df = load_cached_dataframe(old_cache_file)
                if cached_df is not None:
                    d = cached_df.copy()
                    d['source_file'] = os.path.basename(fp)
                    d['Path'] = fp
                    d['Machine No'] = extract_machine_no(fp)
                    all_dfs.append(d)
                    files_loaded_from_cache += 1
                    continue

        # New or modified: needs a Network Drive read.
        files_to_read.append(fp)

    # ---- Pass 2: read every new/changed file in parallel. UNC/SMB reads
    # are network-bound, so a thread pool gives a large speedup over reading
    # one file at a time even with the GIL. ----
    new_records = []

    if files_to_read:
        print(
            f"Reading {len(files_to_read)} new/changed files from the "
            f"Network Drive with {NETWORK_READ_WORKERS} parallel workers..."
        )
        with ThreadPoolExecutor(max_workers=NETWORK_READ_WORKERS) as executor:
            future_to_fp = {
                executor.submit(_read_and_tag, fp, line_name): fp
                for fp in files_to_read
            }
            for future in future_to_fp:
                fp = future_to_fp[future]
                try:
                    _, d = future.result()
                except Exception as e:
                    print(f"  Read error: {fp} -> {e}")
                    continue

                if d is None:
                    continue

                file_size, modified_time = file_infos[fp]
                cache_file = get_disk_cache_path(line_name, log_type, fp)

                # Save a local copy so future requests do not need the
                # Network Drive; batch the DB record instead of writing
                # it one connection at a time.
                if save_cached_dataframe(d, cache_file):
                    new_records.append((
                        line_name, log_type, fp, os.path.basename(fp),
                        file_size, modified_time, cache_file, len(d),
                        datetime.now().isoformat(timespec='seconds')
                    ))

                all_dfs.append(d)
                files_read_from_network += 1

        # ---- Pass 3: one batched write for all new/changed records. ----
        save_cached_file_records_batch(new_records)

    print(
        f"Cache summary: {files_loaded_from_cache} cached, "
        f"{files_read_from_network} network, {len(files)} total files"
    )

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total rows: {len(df)}")
        result = (df, files)
        with cache_lock:
            data_cache[cache_key] = (result, datetime.now())
        return result

    print("No valid data found, generating sample data")
    df = generate_sample_data(line_name, 1000, log_type)
    result = (df, ["[demo data]"])
    with cache_lock:
        data_cache[cache_key] = (result, datetime.now())
    return result

RESERVED_META_COLS = {'source_file', 'path', 'machine no', 'line'}

def detect_columns_retry(df):
    time_col = None
    error_col = None
    lot_col = None

    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in RESERVED_META_COLS:
            continue
        if any(k in col_lower for k in ['time', 'date', 'timestamp', 'วันที่', 'เวลา']):
            if time_col is None:
                time_col = col
        if 'error' in col_lower or 'name' in col_lower or 'description' in col_lower:
            if error_col is None:
                error_col = col
        if 'lot' in col_lower or 'batch' in col_lower:
            if lot_col is None:
                lot_col = col

    if time_col is None and len(df.columns) > 0:
        time_col = df.columns[0]
    if error_col is None and len(df.columns) > 1:
        error_col = df.columns[1]
    if lot_col is None and len(df.columns) > 2:
        lot_col = df.columns[2]

    return time_col, error_col, lot_col

def parse_datetime_robust(series):
    candidates = []
    try:
        candidates.append(pd.to_datetime(series, errors='coerce'))
    except Exception:
        pass
    try:
        candidates.append(pd.to_datetime(series, errors='coerce', dayfirst=True))
    except Exception:
        pass
    for fmt in ('%Y%m%d%H%M%S', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M'):
        try:
            candidates.append(pd.to_datetime(series, errors='coerce', format=fmt))
        except Exception:
            pass
    if not candidates:
        return pd.to_datetime(series, errors='coerce')
    best = max(candidates, key=lambda s: s.notna().sum())
    return best

def detect_columns_cycle(df):
    program_col = None
    cycle_col = None
    lot_col = None
    station_col = None
    date_col = None
    time_col = None

    for col in df.columns:
        col_str = str(col).strip()
        col_lower = col_str.lower()

        if col_lower in RESERVED_META_COLS:
            continue

        if col_lower == 'program name':
            program_col = col
        elif col_lower == 'cycle time':
            cycle_col = col
        elif col_lower == 'production lot':
            lot_col = col
        elif col_lower == 'start date':
            date_col = col
        elif col_lower == 'finish date':
            time_col = col
        elif 'station' in col_lower or 'machine' in col_lower:
            station_col = col

    if program_col is None:
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in RESERVED_META_COLS:
                continue
            if 'program' in col_lower:
                program_col = col
                break

    if cycle_col is None:
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in RESERVED_META_COLS:
                continue
            if 'cycle' in col_lower:
                cycle_col = col
                break

    if lot_col is None:
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in RESERVED_META_COLS:
                continue
            if 'lot' in col_lower:
                lot_col = col
                break

    if date_col is None:
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in RESERVED_META_COLS:
                continue
            if 'start' in col_lower and 'date' in col_lower:
                date_col = col
                break
        if date_col is None:
            for col in df.columns:
                col_lower = str(col).strip().lower()
                if col_lower in RESERVED_META_COLS:
                    continue
                if 'date' in col_lower:
                    date_col = col
                    break

    if time_col is None:
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in RESERVED_META_COLS:
                continue
            if 'finish' in col_lower or 'end' in col_lower:
                time_col = col
                break

    return time_col, date_col, lot_col, program_col, station_col, cycle_col

def detect_columns_solder_paste(df):
    program_col = None
    setup_date_col = None
    finish_date_col = None
    print_ct_ave_col = None
    cleaning_time_col = None
    cleaning_count_col = None

    for col in df.columns:
        col_str = str(col).strip()
        col_lower = col_str.lower()

        if col_lower in RESERVED_META_COLS:
            continue

        if col_lower == 'program name':
            program_col = col
        elif 'setup' in col_lower and 'date' in col_lower:
            setup_date_col = col
        elif 'finish' in col_lower and 'date' in col_lower:
            finish_date_col = col
        elif 'print' in col_lower and 'ct' in col_lower and 'ave' in col_lower:
            print_ct_ave_col = col
        elif 'cleaning' in col_lower and 'time' in col_lower:
            cleaning_time_col = col
        elif 'cleaning' in col_lower and 'count' in col_lower:
            cleaning_count_col = col

    if program_col is None:
        for col in df.columns:
            if 'program' in str(col).lower():
                program_col = col
                break

    if setup_date_col is None:
        for col in df.columns:
            if 'setup' in str(col).lower():
                setup_date_col = col
                break

    if finish_date_col is None:
        for col in df.columns:
            if 'finish' in str(col).lower():
                finish_date_col = col
                break

    return program_col, setup_date_col, finish_date_col, print_ct_ave_col, cleaning_time_col, cleaning_count_col

def process_retry_dataframe(df):
    if df is None or len(df) == 0:
        return None, None, None

    time_col, error_col, lot_col = detect_columns_retry(df)

    if time_col and time_col in df.columns:
        try:
            parsed = parse_datetime_robust(df[time_col])
            df[time_col] = parsed
            df = df.dropna(subset=[time_col])
            if time_col != 'Occurrence Time':
                df.rename(columns={time_col: 'Occurrence Time'}, inplace=True)
        except Exception as e:
            print(f"Time column parse error: {e}")

    if error_col is None or error_col not in df.columns:
        error_col = 'Error Name'
        df[error_col] = 'Unknown Error'

    if lot_col is None or lot_col not in df.columns:
        lot_col = 'Lot No'
        df[lot_col] = 'UNKNOWN'

    return df, error_col, lot_col

def process_cycle_dataframe(df):
    if df is None or len(df) == 0:
        return None, None, None, None, None, None

    time_col, date_col, lot_col, program_col, station_col, cycle_col = detect_columns_cycle(df)

    if date_col and date_col in df.columns:
        try:
            df['DateTime'] = parse_datetime_robust(df[date_col])
        except Exception:
            df['DateTime'] = pd.NaT
    elif time_col and time_col in df.columns:
        try:
            df['DateTime'] = parse_datetime_robust(df[time_col])
        except Exception:
            df['DateTime'] = pd.NaT
    else:
        df['DateTime'] = pd.NaT

    if cycle_col and cycle_col in df.columns:
        df['Cycle Time (s)'] = pd.to_numeric(df[cycle_col], errors='coerce')
    else:
        df['Cycle Time (s)'] = np.nan

    if program_col and program_col in df.columns:
        df['Program Name'] = df[program_col].fillna('UNKNOWN').astype(str).str.strip()
    else:
        df['Program Name'] = 'UNKNOWN'

    if lot_col and lot_col in df.columns:
        df['Lot No'] = df[lot_col].fillna('UNKNOWN').astype(str).str.strip()
    else:
        df['Lot No'] = 'UNKNOWN'

    if station_col and station_col in df.columns:
        df['Station'] = df[station_col].fillna('UNKNOWN').astype(str).str.strip()
    else:
        df['Station'] = 'UNKNOWN'

    df = df.dropna(subset=['DateTime'])
    df = df[df['Cycle Time (s)'].notna() & (df['Cycle Time (s)'] > 0)]

    return df, 'Lot No', 'Program Name', 'Station', 'Cycle Time (s)', 'DateTime'

def process_solder_paste_dataframe(df):
    if df is None or len(df) == 0:
        return None, None, None, None, None, None

    program_col, setup_date_col, finish_date_col, print_ct_ave_col, cleaning_time_col, cleaning_count_col = detect_columns_solder_paste(df)

    if setup_date_col and setup_date_col in df.columns:
        try:
            df['Setup DateTime'] = parse_datetime_robust(df[setup_date_col])
        except Exception:
            df['Setup DateTime'] = pd.NaT
    else:
        df['Setup DateTime'] = pd.NaT

    if finish_date_col and finish_date_col in df.columns:
        try:
            df['Finish DateTime'] = parse_datetime_robust(df[finish_date_col])
        except Exception:
            df['Finish DateTime'] = pd.NaT
    else:
        df['Finish DateTime'] = pd.NaT

    df['Duration (Hours)'] = (df['Finish DateTime'] - df['Setup DateTime']).dt.total_seconds() / 3600

    if program_col and program_col in df.columns:
        df['Program Name'] = df[program_col].fillna('UNKNOWN').astype(str).str.strip()
    else:
        df['Program Name'] = 'UNKNOWN'

    if print_ct_ave_col and print_ct_ave_col in df.columns:
        df['Print CT AVE'] = pd.to_numeric(df[print_ct_ave_col], errors='coerce')
    else:
        df['Print CT AVE'] = np.nan

    if cleaning_time_col and cleaning_time_col in df.columns:
        df['Cleaning Time'] = pd.to_numeric(df[cleaning_time_col], errors='coerce')
    else:
        df['Cleaning Time'] = np.nan

    if cleaning_count_col and cleaning_count_col in df.columns:
        df['Cleaning Count'] = pd.to_numeric(df[cleaning_count_col], errors='coerce')
    else:
        df['Cleaning Count'] = np.nan

    df = df.dropna(subset=['Setup DateTime'])

    return df, program_col, setup_date_col, finish_date_col, print_ct_ave_col, cleaning_time_col, cleaning_count_col

def make_detail_rows(df, limit=DETAIL_LIMIT):
    if df is None or len(df) == 0:
        return [], [], 0

    total = len(df)
    sort_col = 'Occurrence Time' if 'Occurrence Time' in df.columns else 'DateTime' if 'DateTime' in df.columns else df.columns[0]

    try:
        sub = df.sort_values(sort_col, ascending=False) if sort_col in df.columns else df
    except Exception:
        sub = df

    if limit is not None:
        sub = sub.head(limit)

    sub = sub.copy()

    priority_cols = [c for c in ['Machine No', 'Path'] if c in sub.columns]
    if priority_cols:
        other_cols = [c for c in sub.columns if c not in priority_cols]
        sub = sub[priority_cols + other_cols]

    for c in sub.columns:
        if pd.api.types.is_datetime64_any_dtype(sub[c]):
            sub[c] = sub[c].dt.strftime('%d/%m/%Y %H:%M:%S')

    cols = list(sub.columns)
    rows = sub.fillna('').astype(str).values.tolist()
    return cols, rows, total

def make_retry_charts(df, error_col, lot_col):
    summary = df[error_col].astype(str).str.strip().value_counts().head(20).reset_index()
    summary.columns = ['Error Name', 'Count']
    summary['Percent'] = round(summary['Count'] / summary['Count'].sum() * 100, 1)

    DARK_BG = '#0a0e17'
    PANEL_BG = '#141a26'

    bar = px.bar(summary, x='Count', y='Error Name', orientation='h',
                 color='Count', color_continuous_scale=['#3b82f6', '#60a5fa', '#93c5fd'],
                 template='plotly_dark')
    bar.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                      font=dict(family='Inter', color='#e2e8f0'),
                      margin=dict(l=5,r=5,t=5,b=5),
                      coloraxis_showscale=False, height=400)

    pie = px.pie(summary, names='Error Name', values='Count', hole=0.6,
                 color_discrete_sequence=px.colors.sequential.Blues_r, template='plotly_dark')
    pie.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                      font=dict(family='Inter', color='#e2e8f0'),
                      margin=dict(l=5,r=5,t=5,b=5),
                      legend=dict(font=dict(size=9)), height=300)

    if 'Occurrence Time' in df.columns:
        df2 = df.copy()
        df2['Date'] = df2['Occurrence Time'].dt.date
        daily = df2.groupby('Date').size().reset_index(name='Count')
        trend = px.area(daily, x='Date', y='Count', template='plotly_dark',
                        color_discrete_sequence=['#3b82f6'])
        trend.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                             font=dict(family='Inter', color='#e2e8f0'),
                             margin=dict(l=5,r=20,t=5,b=5), height=240)
    else:
        trend = px.line(template='plotly_dark')

    if 'Occurrence Time' in df.columns:
        df3 = df.copy()
        df3['Hour'] = df3['Occurrence Time'].dt.hour
        hourly = (df3.groupby('Hour').size()
                  .reindex(range(24), fill_value=0)
                  .reset_index(name='Count'))
        hourly.columns = ['Hour', 'Count']
        curve = px.line(hourly, x='Hour', y='Count', template='plotly_dark',
                        color_discrete_sequence=['#06b6d4'], markers=True)
        curve.update_traces(line_shape='spline', line_width=3, mode='lines+markers',
                            marker=dict(size=5, color='#06b6d4'),
                            fill='tozeroy', fillcolor='rgba(6,182,212,0.12)')
        curve.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                            font=dict(family='Inter', color='#e2e8f0'),
                            margin=dict(l=5,r=20,t=5,b=5), height=240,
                            xaxis=dict(title='Hour of Day', dtick=2),
                            yaxis=dict(title='Errors'))
    else:
        curve = px.line(template='plotly_dark')

    return summary, bar, pie, trend, curve

def make_cycle_charts(df, lot_col, program_col, station_col, cycle_col):
    DARK_BG = '#0a0e17'
    PANEL_BG = '#141a26'

    if cycle_col not in df.columns or df[cycle_col].isna().all():
        empty = px.line(template='plotly_dark')
        empty.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG)
        return pd.DataFrame(), empty, empty, empty, None

    df_clean = df.dropna(subset=[cycle_col])

    if len(df_clean) == 0:
        empty = px.line(template='plotly_dark')
        empty.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG)
        return pd.DataFrame(), empty, empty, empty, None

    if program_col and program_col in df_clean.columns:
        summary = df_clean.groupby(program_col).agg({
            cycle_col: ['mean', 'min', 'max', 'std', 'count']
        }).round(2).reset_index()
        summary.columns = ['Program', 'Avg (s)', 'Min (s)', 'Max (s)', 'Std Dev', 'Count']
        summary = summary.sort_values('Avg (s)', ascending=False)
    else:
        summary = pd.DataFrame({
            'Metric': ['Total Records', 'Avg Cycle Time', 'Min', 'Max', 'Std Dev'],
            'Value': [len(df_clean), round(df_clean[cycle_col].mean(), 2),
                     round(df_clean[cycle_col].min(), 2),
                     round(df_clean[cycle_col].max(), 2),
                     round(df_clean[cycle_col].std(), 2)]
        })

    if program_col and program_col in df_clean.columns and len(df_clean[program_col].unique()) > 1:
        bar = px.bar(summary, x='Program', y='Avg (s)',
                     error_y='Std Dev', template='plotly_dark',
                     color='Avg (s)', color_continuous_scale=['#3b82f6', '#06b6d4', '#10b981'])
        bar.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                          font=dict(family='Inter', color='#e2e8f0'),
                          margin=dict(l=5,r=5,t=30,b=5), height=320)
    else:
        bar = px.histogram(df_clean, x=cycle_col, nbins=30,
                          template='plotly_dark', color_discrete_sequence=['#3b82f6'])
        bar.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                          font=dict(family='Inter', color='#e2e8f0'),
                          margin=dict(l=5,r=5,t=30,b=5), height=320)

    if program_col and program_col in df_clean.columns and len(df_clean[program_col].unique()) > 1:
        box = px.box(df_clean, x=program_col, y=cycle_col, color=program_col,
                     template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Set2)
        box.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                          font=dict(family='Inter', color='#e2e8f0'),
                          margin=dict(l=5,r=5,t=30,b=5), height=320,
                          showlegend=False)
    elif station_col and station_col in df_clean.columns and len(df_clean[station_col].unique()) > 1:
        box = px.box(df_clean, x=station_col, y=cycle_col, color=station_col,
                     template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Set2)
        box.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                          font=dict(family='Inter', color='#e2e8f0'),
                          margin=dict(l=5,r=5,t=30,b=5), height=320,
                          showlegend=False)
    else:
        box = px.histogram(df_clean, x=cycle_col, nbins=30,
                          template='plotly_dark', color_discrete_sequence=['#06b6d4'])
        box.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                          font=dict(family='Inter', color='#e2e8f0'),
                          margin=dict(l=5,r=5,t=30,b=5), height=320)

    if program_col and program_col in df_clean.columns:
        scatter = px.scatter(df_clean, x=cycle_col, y=program_col,
                            color=station_col if station_col and station_col in df_clean.columns else None,
                            template='plotly_dark',
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            opacity=0.7)
        scatter.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                              font=dict(family='Inter', color='#e2e8f0'),
                              margin=dict(l=5,r=5,t=30,b=5), height=320)
    else:
        scatter = px.scatter(df_clean, x=cycle_col, y=df_clean.index,
                            template='plotly_dark',
                            color_discrete_sequence=['#06b6d4'])
        scatter.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                              font=dict(family='Inter', color='#e2e8f0'),
                              margin=dict(l=5,r=5,t=30,b=5), height=320)

    time_col = 'DateTime' if 'DateTime' in df_clean.columns else df_clean.columns[0]
    if time_col in df_clean.columns and pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df2 = df_clean.copy()
        df2['Date'] = df2[time_col].dt.date
        daily_avg = df2.groupby('Date')[cycle_col].agg(['mean', 'std', 'count']).reset_index()
        daily_avg.columns = ['Date', 'Avg Cycle Time', 'Std Dev', 'Count']

        trend = px.line(daily_avg, x='Date', y='Avg Cycle Time',
                       error_y='Std Dev', template='plotly_dark',
                       color_discrete_sequence=['#10b981'])
        trend.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                            font=dict(family='Inter', color='#e2e8f0'),
                            margin=dict(l=5,r=20,t=30,b=5), height=240)
    else:
        trend = px.line(template='plotly_dark')

    return summary, bar, box, scatter, trend

def make_machine_comparison(df, program_col, machine_col, cycle_col):
    DARK_BG = '#0a0e17'
    PANEL_BG = '#141a26'

    empty_fig = px.line(template='plotly_dark')
    empty_fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG)

    if (df is None or len(df) == 0
            or program_col not in df.columns
            or machine_col not in df.columns
            or cycle_col not in df.columns):
        return [], empty_fig

    d = df.dropna(subset=[cycle_col])
    if len(d) == 0 or d[machine_col].nunique() < 1:
        return [], empty_fig

    grouped = d.groupby([program_col, machine_col])[cycle_col].mean().reset_index()
    grouped.columns = ['Program', 'Machine', 'Avg Cycle Time (s)']

    top_programs = d[program_col].value_counts().head(15).index.tolist()
    chart_df = grouped[grouped['Program'].isin(top_programs)]

    fig = px.bar(
        chart_df, x='Program', y='Avg Cycle Time (s)', color='Machine',
        barmode='group', template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(family='Inter', color='#e2e8f0'),
        margin=dict(l=5, r=5, t=30, b=90), height=360,
        xaxis=dict(tickangle=-35),
        legend=dict(font=dict(size=10))
    )

    pivot = grouped.pivot_table(index='Program', columns='Machine', values='Avg Cycle Time (s)')
    machine_cols = sorted(pivot.columns.tolist())
    pivot = pivot.round(2)

    rows = []
    for prog, row in pivot.iterrows():
        row_dict = {'Program': prog}
        for m in machine_cols:
            val = row.get(m)
            row_dict[m] = '' if pd.isna(val) else val
        valid = row.dropna()
        if len(valid) > 0:
            row_dict['Slowest Machine'] = valid.idxmax()
            row_dict['Fastest Machine'] = valid.idxmin()
            row_dict['Gap (s)'] = round(valid.max() - valid.min(), 2)
        else:
            row_dict['Slowest Machine'] = ''
            row_dict['Fastest Machine'] = ''
            row_dict['Gap (s)'] = ''
        rows.append(row_dict)

    rows.sort(key=lambda r: r['Gap (s)'] if isinstance(r['Gap (s)'], (int, float)) else -1, reverse=True)

    return rows, fig

def chart_json(fig):
    if fig is None:
        return {}
    try:
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e:
        print(f"chart_json error: {e}")
        return {}

# ============================================================
#                     MAIN ROUTE
# ============================================================
@app.route("/", methods=["GET","POST"])
def home():
    selected_line = request.form.get("selected_line", "Line 1")
    log_type = request.form.get("log_type", "Retry Log")
    selected_date = request.form.get("selected_date", "")
    cycle_time_type = request.form.get("cycle_time_type", "standard")

    print(f"Request: line={selected_line}, log_type={log_type}, date={selected_date}, cycle_type={cycle_time_type}")

    # ========== CYCLE TIME LOG ==========
    if log_type == "Cycle Time Log":
        print("=== Processing Cycle Time Log ===")
        
        # ถ้าเลือก Printer ให้ใช้ Solder Paste Log
        if cycle_time_type == "printer":
            print("  → Using Solder Paste Log for Printer")
            actual_log_type = "Solder Paste Log"
        else:
            actual_log_type = "Cycle Time Log"

        if selected_date:
            df, files = load_data_from_line(selected_line, actual_log_type, date_filter=selected_date)
        else:
            df, files = load_data_from_line(selected_line, actual_log_type)

        # ประมวลผลตามประเภท
        if cycle_time_type == "printer":
            # ===== PRINTER MODE =====
            df, program_col, setup_date_col, finish_date_col, print_ct_ave_col, cleaning_time_col, cleaning_count_col = process_solder_paste_dataframe(df)
            
            if df is None or len(df) == 0:
                empty = px.line(template='plotly_dark')
                empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26')
                empty_chart = json.dumps(chart_json(empty))
                return render_template_string(
                    HTML_TEMPLATE,
                    rows=[],
                    bar_chart=empty_chart,
                    pie_chart=empty_chart,
                    trend_chart=empty_chart,
                    curve_chart=empty_chart,
                    lines=list(LINE_CONFIG.keys()),
                    current_line=selected_line,
                    log_type=log_type,
                    cycle_time_type=cycle_time_type,
                    selected_date=selected_date,
                    available_dates=get_available_solder_paste_dates(selected_line),
                    total_errors="0",
                    total_lots="0",
                    error_types="0",
                    files_loaded=0,
                    error_list=[],
                    lot_list=[],
                    date_min="",
                    date_max="",
                    data_range="No Data",
                    is_demo=True,
                    detail_cols=[],
                    detail_rows=[],
                    detail_total=0,
                    is_cycle_time=True,
                    is_solder_paste=False,
                    is_printer_mode=True,
                    machine_chart=empty_chart,
                    machine_rows=[],
                    printer_stats={},
                    summary_rows=[]
                )

            is_demo = files == ["[demo data]"]
            available_dates = get_available_solder_paste_dates(selected_line)

            if 'Setup DateTime' in df.columns and len(df) > 0:
                date_min = df['Setup DateTime'].min().strftime('%Y-%m-%d')
                date_max = df['Setup DateTime'].max().strftime('%Y-%m-%d')
                data_range = f"{df['Setup DateTime'].min().strftime('%d/%m/%Y')} – {df['Setup DateTime'].max().strftime('%d/%m/%Y')}"
            else:
                date_min = date_max = data_range = "N/A"

            # ===== สร้างสถิติ Printer =====
            printer_stats = {
                'Avg Print CT': round(df['Print CT AVE'].mean(), 2) if 'Print CT AVE' in df.columns else 0,
                'Min Print CT': round(df['Print CT AVE'].min(), 2) if 'Print CT AVE' in df.columns else 0,
                'Max Print CT': round(df['Print CT AVE'].max(), 2) if 'Print CT AVE' in df.columns else 0,
                'Total Cleaning': int(df['Cleaning Count'].sum()) if 'Cleaning Count' in df.columns else 0,
                'Avg Cleaning Time': round(df['Cleaning Time'].mean(), 2) if 'Cleaning Time' in df.columns else 0,
                'Total Records': len(df),
                'Programs': df[program_col].nunique() if program_col and program_col in df.columns else 0
            }

            # ===== Bar Chart: Print CT by Program =====
            if program_col and program_col in df.columns:
                summary_chart = df.groupby(program_col).agg({
                    'Print CT AVE': 'mean',
                    'Cleaning Count': 'sum'
                }).reset_index()
                summary_chart = summary_chart.sort_values('Print CT AVE', ascending=False)
                
                bar = px.bar(summary_chart, x='Program Name', y='Print CT AVE',
                            template='plotly_dark',
                            color='Print CT AVE',
                            color_continuous_scale=['#3b82f6', '#06b6d4', '#10b981'],
                            labels={'Print CT AVE': 'Avg Print CT (s)'})
                bar.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                                 font=dict(family='Inter', color='#e2e8f0'),
                                 margin=dict(l=5,r=5,t=30,b=50), height=320,
                                 xaxis=dict(tickangle=-35))
            else:
                bar = px.histogram(df, x='Print CT AVE', nbins=20,
                                  template='plotly_dark',
                                  color_discrete_sequence=['#3b82f6'])
                bar.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                                 font=dict(family='Inter', color='#e2e8f0'),
                                 margin=dict(l=5,r=5,t=30,b=5), height=320)

            # ===== Pie Chart: Cleaning Count by Program =====
            if cleaning_count_col and program_col and program_col in df.columns:
                pie_data = df.groupby(program_col)[cleaning_count_col].sum().reset_index()
                pie_data.columns = ['Program', 'Cleaning Count']
                pie = px.pie(pie_data, names='Program', values='Cleaning Count',
                            hole=0.6, template='plotly_dark',
                            color_discrete_sequence=px.colors.sequential.Blues_r)
                pie.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                                 font=dict(family='Inter', color='#e2e8f0'),
                                 margin=dict(l=5,r=5,t=5,b=5), height=300)
            else:
                pie = px.pie(template='plotly_dark')

            # ===== Trend: Print CT by Date =====
            if 'Setup DateTime' in df.columns:
                df2 = df.copy()
                df2['Date'] = df2['Setup DateTime'].dt.date
                daily = df2.groupby('Date')['Print CT AVE'].mean().reset_index()
                trend = px.area(daily, x='Date', y='Print CT AVE',
                               template='plotly_dark',
                               color_discrete_sequence=['#3b82f6'],
                               labels={'Print CT AVE': 'Avg Print CT (s)'})
                trend.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                                   font=dict(family='Inter', color='#e2e8f0'),
                                   margin=dict(l=5,r=20,t=5,b=5), height=240)
            else:
                trend = px.line(template='plotly_dark')

            # ===== Curve: Cleaning Time by Program =====
            if cleaning_time_col and program_col and program_col in df.columns:
                curve_data = df.groupby(program_col)[cleaning_time_col].mean().reset_index()
                curve = px.bar(curve_data, x='Program Name', y='Cleaning Time',
                              template='plotly_dark',
                              color='Cleaning Time',
                              color_continuous_scale=['#06b6d4', '#3b82f6', '#8b5cf6'],
                              labels={'Cleaning Time': 'Avg Cleaning Time (s)'})
                curve.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                                   font=dict(family='Inter', color='#e2e8f0'),
                                   margin=dict(l=5,r=5,t=30,b=50), height=240,
                                   xaxis=dict(tickangle=-35))
            else:
                curve = px.line(template='plotly_dark')

            detail_cols, detail_rows, detail_total = make_detail_rows(df)
            summary_rows = df.to_dict(orient='records') if len(df) > 0 else []

            return render_template_string(
                HTML_TEMPLATE,
                rows=summary_rows,
                bar_chart=json.dumps(chart_json(bar)),
                pie_chart=json.dumps(chart_json(pie)),
                trend_chart=json.dumps(chart_json(trend)),
                curve_chart=json.dumps(chart_json(curve)),
                lines=list(LINE_CONFIG.keys()),
                current_line=selected_line,
                log_type=log_type,
                cycle_time_type=cycle_time_type,
                selected_date=selected_date,
                available_dates=available_dates,
                total_errors=f"{len(df):,}",
                total_lots=f"{df[program_col].nunique():,}" if program_col and program_col in df.columns else "0",
                error_types=f"{df['Cleaning Count'].sum():,.0f}" if 'Cleaning Count' in df.columns else "0",
                files_loaded=len(files),
                error_list=sorted(df[program_col].dropna().unique().tolist()) if program_col and program_col in df.columns else [],
                lot_list=[],
                date_min=date_min,
                date_max=date_max,
                data_range=data_range,
                is_demo=is_demo,
                detail_cols=detail_cols,
                detail_rows=detail_rows,
                detail_total=detail_total,
                is_cycle_time=True,
                is_solder_paste=False,
                is_printer_mode=True,
                machine_chart=json.dumps(chart_json(px.line(template='plotly_dark'))),
                machine_rows=[],
                printer_stats=printer_stats,
                summary_rows=summary_rows
            )

        else:
            # ===== STANDARD CYCLE TIME MODE =====
            df, lot_col, program_col, station_col, cycle_col, time_col = process_cycle_dataframe(df)

            if df is None or len(df) == 0:
                print("No Cycle Time data available")
                empty = px.line(template='plotly_dark')
                empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26')
                empty_chart = json.dumps(chart_json(empty))

                return render_template_string(
                    HTML_TEMPLATE,
                    rows=[],
                    bar_chart=empty_chart,
                    pie_chart=empty_chart,
                    trend_chart=empty_chart,
                    curve_chart=empty_chart,
                    lines=list(LINE_CONFIG.keys()),
                    current_line=selected_line,
                    log_type=log_type,
                    cycle_time_type=cycle_time_type,
                    selected_date=selected_date,
                    available_dates=get_available_dates(selected_line),
                    total_errors="0",
                    total_lots="0",
                    error_types="0",
                    files_loaded=0,
                    error_list=[],
                    lot_list=[],
                    date_min="",
                    date_max="",
                    data_range="No Data",
                    is_demo=True,
                    detail_cols=[],
                    detail_rows=[],
                    detail_total=0,
                    is_cycle_time=True,
                    is_solder_paste=False,
                    is_printer_mode=False,
                    machine_chart=empty_chart,
                    machine_rows=[],
                    printer_stats={},
                    summary_rows=[]
                )

            is_demo = files == ["[demo data]"]
            available_dates = get_available_dates(selected_line)

            if 'DateTime' in df.columns and len(df) > 0:
                date_min = df['DateTime'].min().strftime('%Y-%m-%d')
                date_max = df['DateTime'].max().strftime('%Y-%m-%d')
                data_range = f"{df['DateTime'].min().strftime('%d/%m/%Y')} – {df['DateTime'].max().strftime('%d/%m/%Y')}"
            else:
                date_min = date_max = data_range = "N/A"

            summary, chart1, chart2, chart3, chart4 = make_cycle_charts(df, lot_col, program_col, station_col, cycle_col)
            machine_rows, machine_fig = make_machine_comparison(df, program_col, 'Machine No', cycle_col)
            detail_cols, detail_rows, detail_total = make_detail_rows(df)

            if program_col and program_col in df.columns:
                error_list = sorted(df[program_col].dropna().unique().tolist())
            else:
                error_list = []
            if lot_col and lot_col in df.columns:
                lot_list = sorted(df[lot_col].dropna().unique().tolist())
            else:
                lot_list = []

            total_errors = len(df)
            total_lots = df[program_col].nunique() if program_col and program_col in df.columns else 0
            error_types = df[station_col].nunique() if station_col and station_col in df.columns else 0

            return render_template_string(
                HTML_TEMPLATE,
                rows=summary.to_dict(orient="records") if isinstance(summary, pd.DataFrame) and len(summary) > 0 else [],
                bar_chart=json.dumps(chart_json(chart1)),
                pie_chart=json.dumps(chart_json(chart2)),
                trend_chart=json.dumps(chart_json(chart3)),
                curve_chart=json.dumps(chart_json(chart4)),
                lines=list(LINE_CONFIG.keys()),
                current_line=selected_line,
                log_type=log_type,
                cycle_time_type=cycle_time_type,
                selected_date=selected_date,
                available_dates=available_dates,
                total_errors=f"{total_errors:,}",
                total_lots=f"{total_lots:,}",
                error_types=f"{error_types:,}",
                files_loaded=len(files),
                error_list=error_list,
                lot_list=lot_list,
                date_min=date_min,
                date_max=date_max,
                data_range=data_range,
                is_demo=is_demo,
                detail_cols=detail_cols,
                detail_rows=detail_rows,
                detail_total=detail_total,
                is_cycle_time=True,
                is_solder_paste=False,
                is_printer_mode=False,
                machine_chart=json.dumps(chart_json(machine_fig)),
                machine_rows=machine_rows,
                printer_stats={},
                summary_rows=[]
            )

    # ========== SOLDER PASTE LOG (Direct) ==========
    elif log_type == "Solder Paste Log":
        print("=== Processing Solder Paste Log ===")

        if selected_date:
            df, files = load_data_from_line(selected_line, log_type, date_filter=selected_date)
        else:
            df, files = load_data_from_line(selected_line, log_type)

        df, program_col, setup_date_col, finish_date_col, print_ct_ave_col, cleaning_time_col, cleaning_count_col = process_solder_paste_dataframe(df)

        if df is None or len(df) == 0:
            empty = px.line(template='plotly_dark')
            empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26')
            empty_chart = json.dumps(chart_json(empty))

            return render_template_string(
                HTML_TEMPLATE,
                rows=[],
                bar_chart=empty_chart,
                pie_chart=empty_chart,
                trend_chart=empty_chart,
                curve_chart=empty_chart,
                lines=list(LINE_CONFIG.keys()),
                current_line=selected_line,
                log_type=log_type,
                cycle_time_type=cycle_time_type,
                selected_date=selected_date,
                available_dates=get_available_solder_paste_dates(selected_line),
                total_errors="0",
                total_lots="0",
                error_types="0",
                files_loaded=0,
                error_list=[],
                lot_list=[],
                date_min="",
                date_max="",
                data_range="No Data",
                is_demo=True,
                detail_cols=[],
                detail_rows=[],
                detail_total=0,
                is_cycle_time=False,
                is_solder_paste=True,
                is_printer_mode=True,
                machine_chart=empty_chart,
                machine_rows=[],
                printer_stats={},
                summary_rows=[]
            )

        is_demo = files == ["[demo data]"]
        available_dates = get_available_solder_paste_dates(selected_line)

        if 'Setup DateTime' in df.columns and len(df) > 0:
            date_min = df['Setup DateTime'].min().strftime('%Y-%m-%d')
            date_max = df['Setup DateTime'].max().strftime('%Y-%m-%d')
            data_range = f"{df['Setup DateTime'].min().strftime('%d/%m/%Y')} – {df['Setup DateTime'].max().strftime('%d/%m/%Y')}"
        else:
            date_min = date_max = data_range = "N/A"

        # ===== Printer Stats =====
        printer_stats = {
            'Avg Print CT': round(df['Print CT AVE'].mean(), 2) if 'Print CT AVE' in df.columns else 0,
            'Min Print CT': round(df['Print CT AVE'].min(), 2) if 'Print CT AVE' in df.columns else 0,
            'Max Print CT': round(df['Print CT AVE'].max(), 2) if 'Print CT AVE' in df.columns else 0,
            'Total Cleaning': int(df['Cleaning Count'].sum()) if 'Cleaning Count' in df.columns else 0,
            'Avg Cleaning Time': round(df['Cleaning Time'].mean(), 2) if 'Cleaning Time' in df.columns else 0,
            'Total Records': len(df),
            'Programs': df[program_col].nunique() if program_col and program_col in df.columns else 0
        }

        # ===== Charts =====
        if program_col and program_col in df.columns:
            summary_chart = df.groupby(program_col).agg({
                'Print CT AVE': 'mean',
                'Cleaning Count': 'sum'
            }).reset_index()
            summary_chart = summary_chart.sort_values('Print CT AVE', ascending=False)
            bar = px.bar(summary_chart, x='Program Name', y='Print CT AVE',
                        template='plotly_dark',
                        color='Print CT AVE',
                        color_continuous_scale=['#3b82f6', '#06b6d4', '#10b981'],
                        labels={'Print CT AVE': 'Avg Print CT (s)'})
            bar.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                             font=dict(family='Inter', color='#e2e8f0'),
                             margin=dict(l=5,r=5,t=30,b=50), height=320,
                             xaxis=dict(tickangle=-35))
        else:
            bar = px.histogram(df, x='Print CT AVE', nbins=20,
                              template='plotly_dark',
                              color_discrete_sequence=['#3b82f6'])
            bar.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                             font=dict(family='Inter', color='#e2e8f0'),
                             margin=dict(l=5,r=5,t=30,b=5), height=320)

        if cleaning_count_col and program_col and program_col in df.columns:
            pie_data = df.groupby(program_col)[cleaning_count_col].sum().reset_index()
            pie_data.columns = ['Program', 'Cleaning Count']
            pie = px.pie(pie_data, names='Program', values='Cleaning Count',
                        hole=0.6, template='plotly_dark',
                        color_discrete_sequence=px.colors.sequential.Blues_r)
            pie.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                             font=dict(family='Inter', color='#e2e8f0'),
                             margin=dict(l=5,r=5,t=5,b=5), height=300)
        else:
            pie = px.pie(template='plotly_dark')

        if 'Setup DateTime' in df.columns:
            df2 = df.copy()
            df2['Date'] = df2['Setup DateTime'].dt.date
            daily = df2.groupby('Date')['Print CT AVE'].mean().reset_index()
            trend = px.area(daily, x='Date', y='Print CT AVE',
                           template='plotly_dark',
                           color_discrete_sequence=['#3b82f6'],
                           labels={'Print CT AVE': 'Avg Print CT (s)'})
            trend.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                               font=dict(family='Inter', color='#e2e8f0'),
                               margin=dict(l=5,r=20,t=5,b=5), height=240)
        else:
            trend = px.line(template='plotly_dark')

        if cleaning_time_col and program_col and program_col in df.columns:
            curve_data = df.groupby(program_col)[cleaning_time_col].mean().reset_index()
            curve = px.bar(curve_data, x='Program Name', y='Cleaning Time',
                          template='plotly_dark',
                          color='Cleaning Time',
                          color_continuous_scale=['#06b6d4', '#3b82f6', '#8b5cf6'],
                          labels={'Cleaning Time': 'Avg Cleaning Time (s)'})
            curve.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                               font=dict(family='Inter', color='#e2e8f0'),
                               margin=dict(l=5,r=5,t=30,b=50), height=240,
                               xaxis=dict(tickangle=-35))
        else:
            curve = px.line(template='plotly_dark')

        detail_cols, detail_rows, detail_total = make_detail_rows(df)
        summary_rows = df.to_dict(orient='records') if len(df) > 0 else []

        return render_template_string(
            HTML_TEMPLATE,
            rows=summary_rows,
            bar_chart=json.dumps(chart_json(bar)),
            pie_chart=json.dumps(chart_json(pie)),
            trend_chart=json.dumps(chart_json(trend)),
            curve_chart=json.dumps(chart_json(curve)),
            lines=list(LINE_CONFIG.keys()),
            current_line=selected_line,
            log_type=log_type,
            cycle_time_type=cycle_time_type,
            selected_date=selected_date,
            available_dates=available_dates,
            total_errors=f"{len(df):,}",
            total_lots=f"{df[program_col].nunique():,}" if program_col and program_col in df.columns else "0",
            error_types=f"{df['Cleaning Count'].sum():,.0f}" if 'Cleaning Count' in df.columns else "0",
            files_loaded=len(files),
            error_list=sorted(df[program_col].dropna().unique().tolist()) if program_col and program_col in df.columns else [],
            lot_list=[],
            date_min=date_min,
            date_max=date_max,
            data_range=data_range,
            is_demo=is_demo,
            detail_cols=detail_cols,
            detail_rows=detail_rows,
            detail_total=detail_total,
            is_cycle_time=False,
            is_solder_paste=True,
            is_printer_mode=True,
            machine_chart=json.dumps(chart_json(px.line(template='plotly_dark'))),
            machine_rows=[],
            printer_stats=printer_stats,
            summary_rows=summary_rows
        )

    # ========== RETRY / ERROR LOG ==========
    else:
        print(f"=== Processing {log_type} ===")

        df_raw, files = load_data_from_line(selected_line, log_type)
        df, error_col, lot_col = process_retry_dataframe(df_raw)

        if df is None or len(df) == 0:
            empty = px.line(template='plotly_dark')
            empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26')
            empty_chart = json.dumps(chart_json(empty))

            return render_template_string(
                HTML_TEMPLATE,
                rows=[],
                bar_chart=empty_chart,
                pie_chart=empty_chart,
                trend_chart=empty_chart,
                curve_chart=empty_chart,
                lines=list(LINE_CONFIG.keys()),
                current_line=selected_line,
                log_type=log_type,
                cycle_time_type=cycle_time_type,
                selected_date="",
                available_dates=[],
                total_errors="0",
                total_lots="0",
                error_types="0",
                files_loaded=0,
                error_list=[],
                lot_list=[],
                date_min="",
                date_max="",
                data_range="No Data",
                is_demo=True,
                detail_cols=[],
                detail_rows=[],
                detail_total=0,
                is_cycle_time=False,
                is_solder_paste=False,
                is_printer_mode=False,
                machine_chart=empty_chart,
                machine_rows=[],
                printer_stats={},
                summary_rows=[]
            )

        is_demo = files == ["[demo data]"]

        if 'Occurrence Time' in df.columns and len(df) > 0:
            date_min = df['Occurrence Time'].min().strftime('%Y-%m-%d')
            date_max = df['Occurrence Time'].max().strftime('%Y-%m-%d')
            data_range = f"{df['Occurrence Time'].min().strftime('%d/%m/%Y')} – {df['Occurrence Time'].max().strftime('%d/%m/%Y')}"
        else:
            date_min = date_max = data_range = "N/A"

        summary, bar, pie, trend, curve = make_retry_charts(df, error_col, lot_col)
        detail_cols, detail_rows, detail_total = make_detail_rows(df)

        return render_template_string(
            HTML_TEMPLATE,
            rows=summary.to_dict(orient="records"),
            bar_chart=json.dumps(chart_json(bar)),
            pie_chart=json.dumps(chart_json(pie)),
            trend_chart=json.dumps(chart_json(trend)),
            curve_chart=json.dumps(chart_json(curve)),
            lines=list(LINE_CONFIG.keys()),
            current_line=selected_line,
            log_type=log_type,
            cycle_time_type=cycle_time_type,
            selected_date="",
            available_dates=[],
            total_errors=f"{len(df):,}",
            total_lots=f"{df[lot_col].nunique():,}",
            error_types=f"{df[error_col].nunique():,}",
            files_loaded=len(files),
            error_list=sorted(df[error_col].dropna().unique().tolist()),
            lot_list=sorted(df[lot_col].dropna().unique().tolist()),
            date_min=date_min,
            date_max=date_max,
            data_range=data_range,
            is_demo=is_demo,
            detail_cols=detail_cols,
            detail_rows=detail_rows,
            detail_total=detail_total,
            is_cycle_time=False,
            is_solder_paste=False,
            is_printer_mode=False,
            machine_chart=json.dumps(chart_json(px.line(template='plotly_dark'))),
            machine_rows=[],
            printer_stats={},
            summary_rows=[]
        )


@app.route("/filter", methods=["POST"])
def filter_data():
    data = request.json
    selected_line = data.get('line', 'Line 1')
    log_type = data.get('log_type', 'Retry Log')
    error_filter = data.get('error_filter', 'all')
    lot_filter = data.get('lot_filter', 'all')
    date_from = data.get('date_from', '')
    date_to = data.get('date_to', '')

    if log_type == "Solder Paste Log":
        df, files = load_data_from_line(selected_line, log_type)
        df, program_col, setup_date_col, finish_date_col, print_ct_ave_col, cleaning_time_col, cleaning_count_col = process_solder_paste_dataframe(df)

        if df is None or len(df) == 0:
            empty = px.line(template='plotly_dark')
            empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26')
            ej = chart_json(empty)
            return jsonify({
                'bar_chart': ej, 'pie_chart': ej, 'trend_chart': ej, 'curve_chart': ej,
                'total_errors': '0', 'total_lots': '0', 'error_types': '0', 'rows': [],
                'detail_cols': [], 'detail_rows': [], 'detail_total': 0,
                'machine_chart': ej, 'machine_rows': []
            })

        if error_filter != 'all' and program_col and program_col in df.columns:
            df = df[df[program_col] == error_filter]
        if date_from and 'Setup DateTime' in df.columns:
            df = df[df['Setup DateTime'].dt.date >= pd.to_datetime(date_from).date()]
        if date_to and 'Setup DateTime' in df.columns:
            df = df[df['Setup DateTime'].dt.date <= pd.to_datetime(date_to).date()]

        if len(df) == 0:
            empty = px.line(template='plotly_dark')
            empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26')
            ej = chart_json(empty)
            return jsonify({
                'bar_chart': ej, 'pie_chart': ej, 'trend_chart': ej, 'curve_chart': ej,
                'total_errors': '0', 'total_lots': '0', 'error_types': '0', 'rows': [],
                'detail_cols': [], 'detail_rows': [], 'detail_total': 0,
                'machine_chart': ej, 'machine_rows': []
            })

        if program_col and program_col in df.columns:
            summary_chart = df.groupby(program_col).agg({
                'Print CT AVE': 'mean',
                'Cleaning Count': 'sum'
            }).reset_index()
            summary_chart = summary_chart.sort_values('Print CT AVE', ascending=False)
            bar = px.bar(summary_chart, x='Program Name', y='Print CT AVE',
                        template='plotly_dark',
                        color='Print CT AVE',
                        color_continuous_scale=['#3b82f6', '#06b6d4', '#10b981'],
                        labels={'Print CT AVE': 'Avg Print CT (s)'})
            bar.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                             font=dict(family='Inter', color='#e2e8f0'),
                             margin=dict(l=5,r=5,t=30,b=50), height=320,
                             xaxis=dict(tickangle=-35))
        else:
            bar = px.histogram(df, x='Print CT AVE', nbins=20,
                              template='plotly_dark',
                              color_discrete_sequence=['#3b82f6'])
            bar.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                             font=dict(family='Inter', color='#e2e8f0'),
                             margin=dict(l=5,r=5,t=30,b=5), height=320)

        if cleaning_count_col and program_col and program_col in df.columns:
            pie_data = df.groupby(program_col)[cleaning_count_col].sum().reset_index()
            pie_data.columns = ['Program', 'Cleaning Count']
            pie = px.pie(pie_data, names='Program', values='Cleaning Count',
                        hole=0.6, template='plotly_dark',
                        color_discrete_sequence=px.colors.sequential.Blues_r)
            pie.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                             font=dict(family='Inter', color='#e2e8f0'),
                             margin=dict(l=5,r=5,t=5,b=5), height=300)
        else:
            pie = px.pie(template='plotly_dark')

        if 'Setup DateTime' in df.columns:
            df2 = df.copy()
            df2['Date'] = df2['Setup DateTime'].dt.date
            daily = df2.groupby('Date')['Print CT AVE'].mean().reset_index()
            trend = px.area(daily, x='Date', y='Print CT AVE',
                           template='plotly_dark',
                           color_discrete_sequence=['#3b82f6'],
                           labels={'Print CT AVE': 'Avg Print CT (s)'})
            trend.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                               font=dict(family='Inter', color='#e2e8f0'),
                               margin=dict(l=5,r=20,t=5,b=5), height=240)
        else:
            trend = px.line(template='plotly_dark')

        if cleaning_time_col and program_col and program_col in df.columns:
            curve_data = df.groupby(program_col)[cleaning_time_col].mean().reset_index()
            curve = px.bar(curve_data, x='Program Name', y='Cleaning Time',
                          template='plotly_dark',
                          color='Cleaning Time',
                          color_continuous_scale=['#06b6d4', '#3b82f6', '#8b5cf6'],
                          labels={'Cleaning Time': 'Avg Cleaning Time (s)'})
            curve.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26',
                               font=dict(family='Inter', color='#e2e8f0'),
                               margin=dict(l=5,r=5,t=30,b=50), height=240,
                               xaxis=dict(tickangle=-35))
        else:
            curve = px.line(template='plotly_dark')

        detail_cols, detail_rows, detail_total = make_detail_rows(df)
        summary_rows = df.to_dict(orient='records') if len(df) > 0 else []

        return jsonify({
            'bar_chart': chart_json(bar),
            'pie_chart': chart_json(pie),
            'trend_chart': chart_json(trend),
            'curve_chart': chart_json(curve),
            'total_errors': f"{len(df):,}",
            'total_lots': f"{df[program_col].nunique():,}" if program_col and program_col in df.columns else "0",
            'error_types': f"{df['Cleaning Count'].sum():,.0f}" if 'Cleaning Count' in df.columns else "0",
            'rows': summary_rows,
            'detail_cols': detail_cols,
            'detail_rows': detail_rows,
            'detail_total': detail_total,
            'machine_chart': chart_json(px.line(template='plotly_dark')),
            'machine_rows': []
        })

    elif log_type == "Cycle Time Log":
        df, files = load_data_from_line(selected_line, log_type)
        df, lot_col, program_col, station_col, cycle_col, time_col = process_cycle_dataframe(df)

        if df is None or len(df) == 0:
            empty = px.line(template='plotly_dark')
            empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26')
            ej = chart_json(empty)
            return jsonify({
                'bar_chart': ej, 'pie_chart': ej, 'trend_chart': ej, 'curve_chart': ej,
                'total_errors': '0', 'total_lots': '0', 'error_types': '0', 'rows': [],
                'detail_cols': [], 'detail_rows': [], 'detail_total': 0,
                'machine_chart': ej, 'machine_rows': []
            })

        if error_filter != 'all' and program_col and program_col in df.columns:
            df = df[df[program_col] == error_filter]
        if lot_filter != 'all' and lot_col and lot_col in df.columns:
            df = df[df[lot_col] == lot_filter]
        if date_from and 'DateTime' in df.columns:
            df = df[df['DateTime'].dt.date >= pd.to_datetime(date_from).date()]
        if date_to and 'DateTime' in df.columns:
            df = df[df['DateTime'].dt.date <= pd.to_datetime(date_to).date()]

        if len(df) == 0:
            empty = px.line(template='plotly_dark')
            empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26')
            ej = chart_json(empty)
            return jsonify({
                'bar_chart': ej, 'pie_chart': ej, 'trend_chart': ej, 'curve_chart': ej,
                'total_errors': '0', 'total_lots': '0', 'error_types': '0', 'rows': [],
                'detail_cols': [], 'detail_rows': [], 'detail_total': 0,
                'machine_chart': ej, 'machine_rows': []
            })

        summary, chart1, chart2, chart3, chart4 = make_cycle_charts(df, lot_col, program_col, station_col, cycle_col)
        machine_rows, machine_fig = make_machine_comparison(df, program_col, 'Machine No', cycle_col)
        detail_cols, detail_rows, detail_total = make_detail_rows(df)

        total_lots = df[program_col].nunique() if program_col and program_col in df.columns else 0
        error_types = df[station_col].nunique() if station_col and station_col in df.columns else 0

        return jsonify({
            'bar_chart': chart_json(chart1),
            'pie_chart': chart_json(chart2),
            'trend_chart': chart_json(chart3),
            'curve_chart': chart_json(chart4),
            'total_errors': f"{len(df):,}",
            'total_lots': f"{total_lots:,}",
            'error_types': f"{error_types:,}",
            'rows': summary.to_dict(orient='records') if isinstance(summary, pd.DataFrame) and len(summary) > 0 else [],
            'detail_cols': detail_cols,
            'detail_rows': detail_rows,
            'detail_total': detail_total,
            'machine_chart': chart_json(machine_fig),
            'machine_rows': machine_rows,
        })

    else:
        df, _ = load_data_from_line(selected_line, log_type)
        df, error_col, lot_col = process_retry_dataframe(df)

        if df is None or len(df) == 0:
            return jsonify({'error': 'no data'}), 500

        if error_filter != 'all':
            df = df[df[error_col] == error_filter]
        if lot_filter != 'all':
            df = df[df[lot_col] == lot_filter]
        if date_from and 'Occurrence Time' in df.columns:
            df = df[df['Occurrence Time'].dt.date >= pd.to_datetime(date_from).date()]
        if date_to and 'Occurrence Time' in df.columns:
            df = df[df['Occurrence Time'].dt.date <= pd.to_datetime(date_to).date()]

        if len(df) == 0:
            empty = px.line(template='plotly_dark')
            empty.update_layout(paper_bgcolor='#0a0e17', plot_bgcolor='#141a26', font_color='#e2e8f0')
            ej = chart_json(empty)
            return jsonify({
                'bar_chart': ej, 'pie_chart': ej, 'trend_chart': ej, 'curve_chart': ej,
                'total_errors': '0', 'total_lots': '0', 'error_types': '0', 'rows': [],
                'detail_cols': [], 'detail_rows': [], 'detail_total': 0
            })

        summary, bar, pie, trend, curve = make_retry_charts(df, error_col, lot_col)
        detail_cols, detail_rows, detail_total = make_detail_rows(df)
        return jsonify({
            'bar_chart': chart_json(bar),
            'pie_chart': chart_json(pie),
            'trend_chart': chart_json(trend),
            'curve_chart': chart_json(curve),
            'total_errors': f"{len(df):,}",
            'total_lots': f"{df[lot_col].nunique():,}",
            'error_types': f"{df[error_col].nunique():,}",
            'rows': summary.to_dict(orient='records'),
            'detail_cols': detail_cols,
            'detail_rows': detail_rows,
            'detail_total': detail_total,
        })


@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    line = request.form.get("selected_line", "Line 1")
    log_type = request.form.get("log_type", "Retry Log")
    
    error_filter = request.form.get("error_filter", "all")
    lot_filter = request.form.get("lot_filter", "all")
    date_from = request.form.get("date_from", "")
    date_to = request.form.get("date_to", "")
    
    df, _ = load_data_from_line(line, log_type)
    
    if log_type == "Cycle Time Log":
        df, lot_col, program_col, station_col, cycle_col, time_col = process_cycle_dataframe(df)
        df = apply_filters_cycle(df, error_filter, lot_filter, date_from, date_to, 
                                 program_col, lot_col, station_col, cycle_col, time_col)
        buf = generate_cycle_pdf_report(df, log_type, line, lot_col, program_col, station_col, cycle_col)
    elif log_type == "Solder Paste Log":
        df, program_col, setup_date_col, finish_date_col, print_ct_ave_col, cleaning_time_col, cleaning_count_col = process_solder_paste_dataframe(df)
        buf = generate_solder_paste_pdf_report(df, line, program_col, setup_date_col, finish_date_col, 
                                               print_ct_ave_col, cleaning_time_col, cleaning_count_col)
    else:
        df, error_col, lot_col = process_retry_dataframe(df)
        if error_filter != 'all':
            df = df[df[error_col] == error_filter]
        if lot_filter != 'all':
            df = df[df[lot_col] == lot_filter]
        if date_from and 'Occurrence Time' in df.columns:
            df = df[df['Occurrence Time'].dt.date >= pd.to_datetime(date_from).date()]
        if date_to and 'Occurrence Time' in df.columns:
            df = df[df['Occurrence Time'].dt.date <= pd.to_datetime(date_to).date()]
        buf = generate_retry_pdf_report(df, error_col, lot_col, line)
    
    return send_file(buf, as_attachment=True,
                     download_name=f"SMT_{line}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                     mimetype='application/pdf')


@app.route("/export/csv", methods=["POST"])
def export_csv():
    line = request.form.get("selected_line", "Line 1")
    log_type = request.form.get("log_type", "Retry Log")
    df, _ = load_data_from_line(line, log_type)

    if log_type == "Cycle Time Log":
        df, lot_col, program_col, station_col, cycle_col, time_col = process_cycle_dataframe(df)
    elif log_type == "Solder Paste Log":
        df, program_col, setup_date_col, finish_date_col, print_ct_ave_col, cleaning_time_col, cleaning_count_col = process_solder_paste_dataframe(df)
    else:
        df, _, _ = process_retry_dataframe(df)

    out = io.BytesIO()
    df.to_csv(out, index=False, encoding='utf-8')
    out.seek(0)
    return send_file(out, as_attachment=True,
                     download_name=f"SMT_{line}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                     mimetype='text/csv')


def apply_filters_cycle(df, error_filter, lot_filter, date_from, date_to, 
                        program_col, lot_col, station_col, cycle_col, time_col):
    if df is None or len(df) == 0:
        return df
    if error_filter != 'all' and program_col and program_col in df.columns:
        df = df[df[program_col] == error_filter]
    if lot_filter != 'all' and lot_col and lot_col in df.columns:
        df = df[df[lot_col] == lot_filter]
    if date_from and 'DateTime' in df.columns:
        df = df[df['DateTime'].dt.date >= pd.to_datetime(date_from).date()]
    if date_to and 'DateTime' in df.columns:
        df = df[df['DateTime'].dt.date <= pd.to_datetime(date_to).date()]
    return df


def generate_retry_pdf_report(df, error_col, lot_col, line_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=18, spaceAfter=12, textColor=colors.HexColor('#3b82f6'))
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=8, textColor=colors.HexColor('#1e40af'))
    normal_style = ParagraphStyle('N', parent=styles['Normal'], fontSize=9, spaceAfter=4)
    story = []
    story.append(Paragraph(f"SMT Error Report — {line_name}", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 12))

    if 'Occurrence Time' in df.columns and len(df) > 0:
        df = df.sort_values('Occurrence Time', ascending=False)

    summary_data = [
        ['Metric', 'Value'],
        ['Total Errors', f"{len(df):,}"],
        ['Total Lots', f"{df[lot_col].nunique():,}"],
        ['Error Types', f"{df[error_col].nunique():,}"],
    ]
    t = Table(summary_data, colWidths=[120, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Top 20 Errors", h2_style))
    ec = df[error_col].value_counts().head(20)
    rows = [['#', 'Error Name', 'Count', '%']]
    for i, (n, c) in enumerate(ec.items(), 1):
        rows.append([str(i), str(n)[:40], f"{c:,}", f"{c/len(df)*100:.1f}%"])
    et = Table(rows, colWidths=[30, 250, 60, 50])
    et.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
    ]))
    story.append(et)

    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_solder_paste_pdf_report(df, line_name, program_col, setup_date_col, finish_date_col,
                                     print_ct_ave_col, cleaning_time_col, cleaning_count_col):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=18, spaceAfter=12, textColor=colors.HexColor('#3b82f6'))
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=8, textColor=colors.HexColor('#1e40af'))
    normal_style = ParagraphStyle('N', parent=styles['Normal'], fontSize=9, spaceAfter=4)
    story = []
    story.append(Paragraph(f"SMT Solder Paste Report — {line_name}", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 12))

    if 'Setup DateTime' in df.columns and len(df) > 0:
        df = df.sort_values('Setup DateTime', ascending=False)

    avg_duration = df['Duration (Hours)'].mean() if 'Duration (Hours)' in df.columns else 0
    total_cleaning = df['Cleaning Count'].sum() if 'Cleaning Count' in df.columns else 0
    avg_print_ct = df['Print CT AVE'].mean() if 'Print CT AVE' in df.columns else 0

    summary_data = [
        ['Metric', 'Value'],
        ['Total Records', f"{len(df):,}"],
        ['Programs', f"{df[program_col].nunique():,}" if program_col and program_col in df.columns else 'N/A'],
        ['Avg Duration (Hours)', f"{avg_duration:.2f}"],
        ['Total Cleaning Count', f"{total_cleaning:.0f}"],
        ['Avg Print CT AVE', f"{avg_print_ct:.2f}"],
    ]
    t = Table(summary_data, colWidths=[120, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Detail Records", h2_style))
    cols_to_show = [program_col, setup_date_col, finish_date_col, 'Duration (Hours)', 
                    print_ct_ave_col, cleaning_time_col, cleaning_count_col, 'Machine No']
    cols_to_show = [c for c in cols_to_show if c and c in df.columns]
    
    table_data = [cols_to_show]
    for _, row in df.head(100).iterrows():
        table_data.append([str(row.get(c, ''))[:30] for c in cols_to_show])
    
    if table_data:
        col_widths = [80] * len(cols_to_show)
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
        ]))
        story.append(t)

    doc.build(story)
    buffer.seek(0)
    return buffer


import matplotlib.pyplot as plt
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image


def generate_cycle_pdf_report(df, log_type, line_name, lot_col, program_col, station_col, cycle_col):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=18, spaceAfter=12, textColor=colors.HexColor('#3b82f6'))
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=8, textColor=colors.HexColor('#1e40af'))
    normal_style = ParagraphStyle('N', parent=styles['Normal'], fontSize=9, spaceAfter=4)
    story = []
    story.append(Paragraph(f"SMT Cycle Time Report — {line_name}", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 12))

    if 'DateTime' in df.columns and len(df) > 0:
        df = df.sort_values('DateTime', ascending=False)
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Records', f"{len(df):,}"],
        ['Programs', f"{df[program_col].nunique():,}" if program_col and program_col in df.columns else 'N/A'],
        ['Stations', f"{df[station_col].nunique():,}" if station_col and station_col in df.columns else 'N/A'],
        ['Avg Cycle Time', f"{df[cycle_col].mean():.2f}s" if cycle_col and cycle_col in df.columns else 'N/A'],
        ['Min Cycle Time', f"{df[cycle_col].min():.2f}s" if cycle_col and cycle_col in df.columns else 'N/A'],
        ['Max Cycle Time', f"{df[cycle_col].max():.2f}s" if cycle_col and cycle_col in df.columns else 'N/A'],
    ]
    t = Table(summary_data, colWidths=[120, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    if (program_col and program_col in df.columns 
        and 'Machine No' in df.columns 
        and cycle_col and cycle_col in df.columns):
        
        story.append(Paragraph("Machine Comparison by Program", h2_style))
        story.append(Spacer(1, 8))
        
        d = df.dropna(subset=[cycle_col])
        if len(d) > 0 and d['Machine No'].nunique() > 1:
            grouped = d.groupby([program_col, 'Machine No'])[cycle_col].mean().reset_index()
            grouped.columns = ['Program', 'Machine', 'Avg Cycle Time (s)']
            
            top_programs = d[program_col].value_counts().head(1000).index.tolist()
            chart_df = grouped[grouped['Program'].isin(top_programs)]
            
            if len(chart_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#0a0e17')
                ax.set_facecolor('#141a26')
                
                pivot = chart_df.pivot(index='Program', columns='Machine', values='Avg Cycle Time (s)')
                pivot = pivot.sort_values(pivot.columns[0] if len(pivot.columns) > 0 else None, ascending=False)
                
                pivot.plot(kind='bar', ax=ax, width=0.8)
                
                ax.set_title('Avg Cycle Time by Program & Machine', color='#e2e8f0', fontsize=14, fontweight='bold')
                ax.set_xlabel('Program', color='#94a3b8', fontsize=10)
                ax.set_ylabel('Avg Cycle Time (s)', color='#94a3b8', fontsize=10)
                ax.tick_params(colors='#94a3b8', labelsize=9)
                ax.tick_params(axis='x', rotation=35, labelsize=8)
                
                legend = ax.legend(loc='upper right', facecolor='#141a26', edgecolor='#2a3548')
                for text in legend.get_texts():
                    text.set_color('#e2e8f0')
                
                ax.grid(axis='y', alpha=0.2, color='#2a3548')
                ax.set_axisbelow(True)
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f', fontsize=7, color='#e2e8f0', padding=1)
                
                plt.tight_layout()
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0e17')
                plt.close()
                img_buffer.seek(0)
                
                img = Image(img_buffer, width=420, height=250)
                story.append(img)
                story.append(Spacer(1, 12))
                
                pivot_table = pivot.reset_index()
                pivot_table = pivot_table.round(2)
                
                table_rows = [['Program'] + [str(m) for m in pivot.columns.tolist()] + ['Slowest', 'Fastest', 'Gap (s)']]
                
                for prog in pivot.index:
                    row = [str(prog)]
                    values = []
                    for m in pivot.columns:
                        val = pivot.loc[prog, m]
                        row.append(f"{val:.2f}" if not pd.isna(val) else '-')
                        if not pd.isna(val):
                            values.append((m, val))
                    
                    if values:
                        slowest = max(values, key=lambda x: x[1])
                        fastest = min(values, key=lambda x: x[1])
                        row.append(slowest[0])
                        row.append(fastest[0])
                        row.append(f"{slowest[1] - fastest[1]:.2f}")
                    else:
                        row.append('-')
                        row.append('-')
                        row.append('-')
                    table_rows.append(row)
                
                col_widths = [80] + [50] * len(pivot.columns) + [60, 60, 50]
                t = Table(table_rows, colWidths=col_widths)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3b82f6')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 8),
                    ('FONTSIZE', (0,1), (-1,-1), 7),
                    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2a3548')),
                    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#141a26')),
                    ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#e2e8f0')),
                ]))
                story.append(t)
                story.append(Spacer(1, 16))

    story.append(Paragraph("Top Programs by Average Cycle Time", h2_style))
    if program_col and program_col in df.columns and cycle_col and cycle_col in df.columns:
        top_programs = df.groupby(program_col)[cycle_col].mean().sort_values(ascending=False).head(500)
        rows = [['#', 'Program', 'Avg Cycle Time (s)', 'Count']]
        for i, (prog, avg) in enumerate(top_programs.items(), 1):
            count = len(df[df[program_col] == prog])
            rows.append([str(i), str(prog)[:40], f"{avg:.2f}", f"{count:,}"])
        et = Table(rows, colWidths=[30, 200, 70, 50])
        et.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2a3548')),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#141a26')),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#e2e8f0')),
        ]))
        story.append(et)

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
#                     HTML TEMPLATE
# ============================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMT Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
:root {
  --bg: #0a0e17;
  --surface: #141a26;
  --surface2: #1a2333;
  --surface3: #1f2a3c;
  --border: #2a3548;
  --border2: #3a4a62;
  --blue: #3b82f6;
  --cyan: #06b6d4;
  --green: #10b981;
  --orange: #f59e0b;
  --red: #ef4444;
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --gradient-1: linear-gradient(135deg, #3b82f6, #06b6d4);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  line-height: 1.5;
  overflow: hidden;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

.app { display: flex; height: 100vh; overflow: hidden; }

.sidebar {
  width: 280px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  backdrop-filter: blur(10px);
}
.sidebar-header {
  padding: 24px 20px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(6,182,212,0.1));
}
.logo { display: flex; align-items: center; gap: 12px; }
.logo-icon { font-size: 32px; }
.logo-text h1 {
  font-size: 18px;
  font-weight: 700;
  background: var(--gradient-1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.logo-text p { font-size: 10px; color: var(--text-muted); margin-top: 2px; }

.sidebar-nav { padding: 20px; }
.nav-item {
  padding: 10px 12px;
  margin-bottom: 8px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--text-muted);
  transition: all 0.2s;
}
.nav-item:hover { background: var(--surface2); color: var(--text); }
.nav-item.active { background: var(--surface3); color: var(--blue); border-left: 3px solid var(--blue); }

.sidebar-section { padding: 16px 20px 8px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); }

.form-group { padding: 0 16px; margin-bottom: 16px; }
.form-label { font-size: 11px; color: var(--text-muted); margin-bottom: 6px; display: block; font-weight: 500; }
.form-select {
  width: 100%;
  background: var(--surface2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
}
.form-select:focus { outline: none; border-color: var(--blue); box-shadow: 0 0 0 3px rgba(59,130,246,0.1); }

.btn {
  width: 100%;
  padding: 10px;
  border-radius: 8px;
  border: none;
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 8px;
}
.btn-primary { background: var(--gradient-1); color: white; }
.btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.4); }
.btn-outline { background: transparent; color: var(--text-muted); border: 1px solid var(--border); }
.btn-outline:hover { background: var(--surface2); color: var(--text); border-color: var(--border2); }

.system-status { margin: 16px; padding: 12px; background: var(--surface2); border-radius: 8px; border: 1px solid var(--border); }
.status-item { display: flex; justify-content: space-between; padding: 6px 0; }
.status-label { font-size: 11px; color: var(--text-dim); }
.status-value { font-size: 11px; color: var(--text-muted); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.main { flex: 1; overflow-y: auto; overflow-x: hidden; }
.top-bar {
  padding: 16px 24px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
  backdrop-filter: blur(10px);
}
.top-bar h2 { font-size: 20px; font-weight: 600; }
.top-bar p { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
.content { padding: 20px 24px; }

.filter-bar {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 20px;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  align-items: flex-end;
}
.filter-group { flex: 1; min-width: 150px; }
.filter-group label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-dim); display: block; margin-bottom: 6px; }
.filter-select, .filter-date {
  width: 100%;
  background: var(--surface2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12px;
}
.filter-btn {
  padding: 8px 20px;
  background: var(--gradient-1);
  border: none;
  border-radius: 8px;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.filter-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.3); }

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  position: relative;
  transition: all 0.3s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: var(--border2); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--gradient-1); }
.kpi-icon { font-size: 28px; margin-bottom: 12px; }
.kpi-value { font-size: 32px; font-weight: 700; font-family: 'Inter', monospace; line-height: 1; }
.kpi-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.7px; margin-top: 8px; }

.charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.charts-row-2 { display: grid; grid-template-columns: 1.6fr 1fr; gap: 16px; margin-bottom: 16px; }
.chart-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.chart-header { padding: 12px 16px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 13px; display: flex; align-items: center; gap: 8px; }
.chart-body { padding: 12px; }

.table-container { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 16px; }
.table-header { padding: 12px 16px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
.table-header .row-count { font-size: 11px; font-weight: 500; color: var(--text-dim); }
.table-scroll { overflow-x: auto; max-height: 400px; overflow-y: auto; }
.table-scroll-detail { overflow: auto; max-height: 420px; }

.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.data-table th {
  padding: 10px 12px;
  text-align: left;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: var(--text-dim);
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 1;
}
.data-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
.data-table tr:hover td { background: var(--surface2); }

.loading-overlay {
  position: fixed;
  inset: 0;
  background: rgba(10,14,23,0.95);
  backdrop-filter: blur(8px);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  flex-direction: column;
  gap: 16px;
}
.loading-overlay.active { display: flex; }
.spinner {
  width: 48px;
  height: 48px;
  border: 3px solid var(--border);
  border-top-color: var(--blue);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 1024px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .charts-row, .charts-row-2 { grid-template-columns: 1fr; }
  .sidebar { width: 260px; }
}
</style>
</head>
<body>

<div class="loading-overlay" id="loadingOverlay">
  <div class="spinner"></div>
  <div style="font-size: 14px;">Loading data...</div>
</div>

<div class="app">
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="logo">
        <div class="logo-icon">⚡</div>
        <div class="logo-text">
          <h1>SMT Intel</h1>
          <p>Manufacturing Analytics</p>
        </div>
      </div>
    </div>

    <div class="sidebar-nav">
      <div class="nav-item active"><span>📊</span> Dashboard</div>
      <div class="nav-item"><span>📈</span> Analytics</div>
      <div class="nav-item"><span>📁</span> Reports</div>
    </div>

    <div class="sidebar-section">Configuration</div>

    <form method="POST" id="mainForm">
      <div class="form-group">
        <label class="form-label">Log Type</label>
        <select class="form-select" name="log_type" id="logTypeSelect" onchange="toggleDateFilter()">
          <option value="Retry Log" {% if log_type=='Retry Log' %}selected{% endif %}>📋 Retry Log</option>
          <option value="Error Log" {% if log_type=='Error Log' %}selected{% endif %}>🔴 Error Log</option>
          <option value="Cycle Time Log" {% if log_type=='Cycle Time Log' %}selected{% endif %}>⏱️ Cycle Time Log</option>
          <option value="Solder Paste Log" {% if log_type=='Solder Paste Log' %}selected{% endif %}>🖨️ Solder Paste Log</option>
        </select>
      </div>

      <div class="form-group" id="cycleTimeTypeGroup" style="{% if log_type != 'Cycle Time Log' %}display:none;{% endif %}">
        <label class="form-label">Cycle Time Type</label>
        <select class="form-select" name="cycle_time_type" id="cycleTimeTypeSelect">
          <option value="standard" {% if cycle_time_type=='standard' %}selected{% endif %}>⏱️ Standard Cycle Time</option>
          <option value="printer" {% if cycle_time_type=='printer' %}selected{% endif %}>🖨️ Printer (Solder Paste)</option>
        </select>
      </div>

      <div class="form-group" id="dateFilterGroup" style="{% if log_type != 'Cycle Time Log' and log_type != 'Solder Paste Log' %}display:none;{% endif %}">
        <label class="form-label">Select Date</label>
        <select class="form-select" name="selected_date">
          <option value="">All Dates</option>
          {% for date in available_dates %}
          <option value="{{ date.raw }}" {% if date.raw==selected_date %}selected{% endif %}>
            {{ date.display }}
          </option>
          {% endfor %}
        </select>
      </div>

      <div class="form-group">
        <label class="form-label">Production Line</label>
        <select class="form-select" name="selected_line" id="lineSelect">
          {% for line in lines %}
          <option value="{{ line }}" {% if line==current_line %}selected{% endif %}>{{ line }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="form-group">
        <button class="btn btn-primary" type="submit">⟳ Load Data</button>
      </div>
    </form>

    <form method="POST" action="/export/pdf" target="_blank" id="pdfForm">
      <input type="hidden" name="selected_line" value="{{ current_line }}">
      <input type="hidden" name="log_type" value="{{ log_type }}">
      <input type="hidden" name="error_filter" id="pdfErrorFilter" value="all">
      <input type="hidden" name="lot_filter" id="pdfLotFilter" value="all">
      <input type="hidden" name="date_from" id="pdfDateFrom" value="">
      <input type="hidden" name="date_to" id="pdfDateTo" value="">
      <div class="form-group">
        <button class="btn btn-outline" type="submit">📑 Export PDF</button>
      </div>
    </form>

    <form method="POST" action="/export/csv" id="csvForm">
      <input type="hidden" name="selected_line" value="{{ current_line }}">
      <input type="hidden" name="log_type" value="{{ log_type }}">
      <div class="form-group">
        <button class="btn btn-outline" type="submit">📊 Export CSV</button>
      </div>
    </form>

    <div class="system-status">
      <div class="status-item"><span class="status-label">System Status</span><span class="status-value"><span class="status-dot"></span> Online</span></div>
      <div class="status-item"><span class="status-label">Current Line</span><span class="status-value">{{ current_line }}</span></div>
      <div class="status-item"><span class="status-label">Files Loaded</span><span class="status-value">{{ files_loaded }}</span></div>
    </div>
  </aside>

  <main class="main">
    <div class="top-bar">
      <h2>
        {% if is_printer_mode %}
          🖨️ Printer Dashboard
        {% elif is_cycle_time %}
          ⏱️ Cycle Time Dashboard
        {% elif is_solder_paste %}
          🖨️ Solder Paste Dashboard
        {% else %}
          📊 Error Intelligence Dashboard
        {% endif %}
      </h2>
      <p>{{ current_line }} · {{ log_type }} · {{ data_range }}</p>
    </div>

    <div class="content">
      {% if is_demo %}
      <div style="background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px; color: #fbbf24;">
        ⚠ Demo mode - No data files found
      </div>
      {% endif %}

      <!-- ===== PRINTER KPI - เฉพาะ Printer Mode ===== -->
      {% if is_printer_mode and printer_stats %}
      <div class="kpi-grid" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 20px;">
        <div class="kpi-card" style="border-color: #3b82f6;">
          <div class="kpi-icon">🖨️</div>
          <div class="kpi-value">{{ printer_stats.get('Avg Print CT', 0) }}s</div>
          <div class="kpi-label">Avg Print CT</div>
        </div>
        <div class="kpi-card" style="border-color: #f59e0b;">
          <div class="kpi-icon">📊</div>
          <div class="kpi-value">{{ printer_stats.get('Min Print CT', 0) }} - {{ printer_stats.get('Max Print CT', 0) }}s</div>
          <div class="kpi-label">Min / Max</div>
        </div>
        <div class="kpi-card" style="border-color: #10b981;">
          <div class="kpi-icon">🧹</div>
          <div class="kpi-value">{{ printer_stats.get('Total Cleaning', 0) }}</div>
          <div class="kpi-label">Total Cleaning</div>
        </div>
        <div class="kpi-card" style="border-color: #8b5cf6;">
          <div class="kpi-icon">⏱️</div>
          <div class="kpi-value">{{ printer_stats.get('Avg Cleaning Time', 0) }}s</div>
          <div class="kpi-label">Avg Cleaning Time</div>
        </div>
      </div>
      {% endif %}

      <!-- Filter Bar -->
      <div class="filter-bar">
        <div class="filter-group">
          <label>{% if is_cycle_time or is_printer_mode or is_solder_paste %}Program{% else %}Error Type{% endif %}</label>
          <select class="filter-select" id="filterError">
            <option value="all">All</option>
            {% for e in error_list %}<option value="{{ e }}">{{ e }}</option>{% endfor %}
          </select>
        </div>
        <div class="filter-group">
          <label>Lot</label>
          <select class="filter-select" id="filterLot">
            <option value="all">All Lots</option>
            {% for l in lot_list %}<option value="{{ l }}">{{ l }}</option>{% endfor %}
          </select>
        </div>
        <div class="filter-group">
          <label>Date From</label>
          <input type="date" class="filter-date" id="dateFrom" value="{{ date_min }}">
        </div>
        <div class="filter-group">
          <label>Date To</label>
          <input type="date" class="filter-date" id="dateTo" value="{{ date_max }}">
        </div>
        <button class="filter-btn" onclick="applyFilters()">Apply Filters</button>
        <button class="filter-btn" onclick="clearFilters()" style="background: var(--surface2); color: var(--text-muted);">Clear</button>
      </div>

      <!-- Main KPI Grid -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-icon">{% if is_printer_mode %}🖨️{% elif is_cycle_time %}⏱️{% elif is_solder_paste %}🖨️{% else %}⚠️{% endif %}</div>
          <div class="kpi-value" id="kpiErrors">{{ total_errors }}</div>
          <div class="kpi-label">{% if is_printer_mode or is_solder_paste %}Total Records{% elif is_cycle_time %}Total Records{% else %}Total Errors{% endif %}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-icon">📦</div>
          <div class="kpi-value" id="kpiLots">{{ total_lots }}</div>
          <div class="kpi-label">{% if is_printer_mode or is_solder_paste %}Programs{% elif is_cycle_time %}Programs{% else %}Lots Affected{% endif %}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-icon">🔧</div>
          <div class="kpi-value" id="kpiTypes">{{ error_types }}</div>
          <div class="kpi-label">{% if is_printer_mode or is_solder_paste %}Cleaning Count{% elif is_cycle_time %}Stations{% else %}Error Types{% endif %}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-icon">📁</div>
          <div class="kpi-value">{{ files_loaded }}</div>
          <div class="kpi-label">Files Loaded</div>
        </div>
      </div>

      <!-- Charts Row -->
      <div class="charts-row">
        <div class="chart-panel">
          <div class="chart-header">
            <span>📊</span> 
            {% if is_printer_mode or is_solder_paste %}
              Print CT by Program
            {% elif is_cycle_time %}
              Average Cycle Time by Program
            {% else %}
              Top Error Types
            {% endif %}
          </div>
          <div class="chart-body">
            <div id="chartBar" style="height: 320px;"></div>
          </div>
        </div>
        <div class="chart-panel">
          <div class="chart-header">
            <span>📊</span> 
            {% if is_printer_mode or is_solder_paste %}
              Cleaning Count Distribution
            {% elif is_cycle_time %}
              Cycle Time Distribution
            {% else %}
              Error Distribution
            {% endif %}
          </div>
          <div class="chart-body">
            <div id="chartPie" style="height: 320px;"></div>
          </div>
        </div>
      </div>

      <!-- Charts Row 2 -->
      <div class="charts-row-2">
        <div class="chart-panel">
          <div class="chart-header">
            <span>📈</span> 
            {% if is_printer_mode or is_solder_paste %}
              Print CT Trend
            {% elif is_cycle_time %}
              Daily Average Cycle Time
            {% else %}
              Daily Error Trend
            {% endif %}
          </div>
          <div class="chart-body">
            <div id="chartTrend" style="height: 260px;"></div>
          </div>
        </div>
        <div class="chart-panel">
          <div class="chart-header">
            <span>📉</span> 
            {% if is_printer_mode or is_solder_paste %}
              Cleaning Time by Program
            {% elif is_cycle_time %}
              Cycle Time Scatter
            {% else %}
              Errors by Hour
            {% endif %}
          </div>
          <div class="chart-body">
            <div id="chartCurve" style="height: 260px;"></div>
          </div>
        </div>
      </div>

      <!-- Machine Comparison (เฉพาะ Cycle Time Standard) -->
      {% if is_cycle_time and not is_printer_mode %}
      <div class="chart-panel" style="margin-bottom: 16px;">
        <div class="chart-header">
          <span>🏭</span> เปรียบเทียบ Cycle Time ตาม Machine No
        </div>
        <div class="chart-body">
          <div id="chartMachine" style="height: 360px;"></div>
        </div>
      </div>

      <div class="table-container">
        <div class="table-header">
          🏭 Machine Comparison by Program
          <span class="row-count" id="machineRowCount">{{ machine_rows|length }} programs</span>
        </div>
        <div class="table-scroll">
          <table class="data-table" id="machineTable">
            <thead id="machineHead">
              {% if machine_rows %}
              <tr>
                {% for key in machine_rows[0].keys() %}
                <th>{{ key }}</th>
                {% endfor %}
              </tr>
              {% endif %}
            </thead>
            <tbody id="machineBody">
              {% for row in machine_rows %}
              <tr>
                {% for value in row.values() %}
                <td>{{ value }}</td>
                {% endfor %}
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
      {% endif %}

      <!-- Summary Table -->
      <div class="table-container">
        <div class="table-header">
          📌 {% if is_printer_mode or is_solder_paste %}Printer Detail{% elif is_cycle_time %}Cycle Time Summary{% else %}Error Summary{% endif %}
          <span class="row-count">{{ rows|length }} records</span>
        </div>
        <div class="table-scroll">
          <table class="data-table" id="summaryTable">
            <thead id="summaryHead">
              {% if rows %}
              <tr>
                {% for key in rows[0].keys() %}
                <th>{{ key }}</th>
                {% endfor %}
              </tr>
              {% endif %}
            </thead>
            <tbody id="tableBody">
              {% for row in rows %}
              <tr>
                {% for value in row.values() %}
                <td>{{ value }}</td>
                {% endfor %}
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Detail Records -->
      <div class="table-container">
        <div class="table-header">
          <span>🗂️ Detail Records</span>
          <span class="row-count" id="detailCount"></span>
        </div>
        <div class="table-scroll-detail">
          <table class="data-table" id="detailTable">
            <thead><tr id="detailHead"></tr></thead>
            <tbody id="detailBody"></tbody>
          </table>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 16px; border-top:1px solid var(--border);">
          <button onclick="detailPrev()" style="padding:6px 14px; background:var(--surface2); color:var(--text); border:1px solid var(--border); border-radius:8px; cursor:pointer; font-family:'Inter',sans-serif; font-size:12px;">‹ Prev</button>
          <span id="detailPageInfo" class="row-count"></span>
          <button onclick="detailNext()" style="padding:6px 14px; background:var(--surface2); color:var(--text); border:1px solid var(--border); border-radius:8px; cursor:pointer; font-family:'Inter',sans-serif; font-size:12px;">Next ›</button>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
function updateExportFilters() {
  document.getElementById('pdfErrorFilter').value = document.getElementById('filterError').value;
  document.getElementById('pdfLotFilter').value = document.getElementById('filterLot').value;
  document.getElementById('pdfDateFrom').value = document.getElementById('dateFrom').value;
  document.getElementById('pdfDateTo').value = document.getElementById('dateTo').value;
}

document.getElementById('pdfForm').addEventListener('submit', function(e) {
  updateExportFilters();
});

function toggleDateFilter() {
  const logType = document.getElementById('logTypeSelect').value;
  const dateGroup = document.getElementById('dateFilterGroup');
  const cycleTypeGroup = document.getElementById('cycleTimeTypeGroup');
  
  if (logType === 'Cycle Time Log') {
    cycleTypeGroup.style.display = 'block';
  } else {
    cycleTypeGroup.style.display = 'none';
  }
  
  dateGroup.style.display = (logType === 'Cycle Time Log' || logType === 'Solder Paste Log') ? 'block' : 'none';
}

document.addEventListener('DOMContentLoaded', toggleDateFilter);

const BAR_DATA = {{ bar_chart|safe }};
const PIE_DATA = {{ pie_chart|safe }};
const TREND_DATA = {{ trend_chart|safe }};
const CURVE_DATA = {{ curve_chart|safe }};
const MACHINE_DATA = {{ machine_chart|safe }};

const config = { responsive: true, displayModeBar: false };

const DETAIL_PAGE_SIZE = 100;
let DETAIL_COLS = {{ detail_cols|tojson }};
let DETAIL_ROWS = {{ detail_rows|tojson }};
let DETAIL_TOTAL = {{ detail_total }};
let detailPage = 1;

function renderDetailHead() {
  document.getElementById('detailHead').innerHTML =
    '<th>#</th>' + DETAIL_COLS.map(c => `<th>${escapeHtml(c)}</th>`).join('');
}
function renderDetailPage() {
  const pages = Math.max(1, Math.ceil(DETAIL_ROWS.length / DETAIL_PAGE_SIZE));
  if (detailPage > pages) detailPage = pages;
  if (detailPage < 1) detailPage = 1;
  const start = (detailPage - 1) * DETAIL_PAGE_SIZE;
  const slice = DETAIL_ROWS.slice(start, start + DETAIL_PAGE_SIZE);
  let html = '';
  slice.forEach((r, i) => {
    html += `<tr><td>${start+i+1}</td>` + r.map(v => `<td>${escapeHtml(v)}</td>`).join('') + '</tr>';
  });
  document.getElementById('detailBody').innerHTML = html;
  document.getElementById('detailCount').textContent = `Total ${DETAIL_TOTAL.toLocaleString()} records`;
  document.getElementById('detailPageInfo').textContent = `Page ${detailPage} / ${pages}`;
}
function detailPrev(){ detailPage--; renderDetailPage(); }
function detailNext(){ detailPage++; renderDetailPage(); }

function renderMachineTable(rows) {
  const head = document.getElementById('machineHead');
  const body = document.getElementById('machineBody');
  const countEl = document.getElementById('machineRowCount');
  if (!head || !body) return;
  if (!rows || rows.length === 0) {
    head.innerHTML = '';
    body.innerHTML = '';
    if (countEl) countEl.textContent = '0 programs';
    return;
  }
  const headers = Object.keys(rows[0]);
  head.innerHTML = '<tr>' + headers.map(h => `<th>${escapeHtml(h)}</th>`).join('') + '</tr>';
  let html = '';
  rows.forEach(row => {
    html += '<tr>' + headers.map(h => `<td>${escapeHtml(row[h])}</td>`).join('') + '</tr>';
  });
  body.innerHTML = html;
  if (countEl) countEl.textContent = `${rows.length} programs`;
}

function initCharts() {
  if (BAR_DATA && BAR_DATA.data) Plotly.newPlot('chartBar', BAR_DATA.data, BAR_DATA.layout, config);
  if (PIE_DATA && PIE_DATA.data) Plotly.newPlot('chartPie', PIE_DATA.data, PIE_DATA.layout, config);
  if (TREND_DATA && TREND_DATA.data) Plotly.newPlot('chartTrend', TREND_DATA.data, TREND_DATA.layout, config);
  if (CURVE_DATA && CURVE_DATA.data) Plotly.newPlot('chartCurve', CURVE_DATA.data, CURVE_DATA.layout, config);
  if (document.getElementById('chartMachine') && MACHINE_DATA && MACHINE_DATA.data) {
    Plotly.newPlot('chartMachine', MACHINE_DATA.data, MACHINE_DATA.layout, config);
  }
}

function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function applyFilters() {
  const filters = {
    line: '{{ current_line }}',
    log_type: '{{ log_type }}',
    error_filter: document.getElementById('filterError').value,
    lot_filter: document.getElementById('filterLot').value,
    date_from: document.getElementById('dateFrom').value,
    date_to: document.getElementById('dateTo').value
  };

  document.getElementById('loadingOverlay').classList.add('active');

  fetch('/filter', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(filters)
  })
  .then(r => r.json())
  .then(d => {
    if (d.bar_chart && d.bar_chart.data) Plotly.react('chartBar', d.bar_chart.data, d.bar_chart.layout);
    if (d.pie_chart && d.pie_chart.data) Plotly.react('chartPie', d.pie_chart.data, d.pie_chart.layout);
    if (d.trend_chart && d.trend_chart.data) Plotly.react('chartTrend', d.trend_chart.data, d.trend_chart.layout);
    if (d.curve_chart && d.curve_chart.data) Plotly.react('chartCurve', d.curve_chart.data, d.curve_chart.layout);
    if (d.machine_chart && d.machine_chart.data && document.getElementById('chartMachine')) {
      Plotly.react('chartMachine', d.machine_chart.data, d.machine_chart.layout);
    }
    if ('machine_rows' in d) renderMachineTable(d.machine_rows);

    document.getElementById('kpiErrors').textContent = d.total_errors;
    document.getElementById('kpiLots').textContent = d.total_lots;
    document.getElementById('kpiTypes').textContent = d.error_types;

    const tbody = document.getElementById('tableBody');
    const thead = document.getElementById('summaryHead');
    tbody.innerHTML = '';
    if (d.rows && d.rows.length > 0) {
      const headers = Object.keys(d.rows[0]);
      thead.innerHTML = '<tr>' + headers.map(h => `<th>${escapeHtml(h)}</th>`).join('') + '</tr>';
      d.rows.forEach((row) => {
        tbody.innerHTML += '<tr>' + headers.map(h => `<td>${escapeHtml(row[h])}</td>`).join('') + '</tr>';
      });
    } else {
      thead.innerHTML = '';
    }

    DETAIL_COLS = d.detail_cols || [];
    DETAIL_ROWS = d.detail_rows || [];
    DETAIL_TOTAL = d.detail_total || 0;
    detailPage = 1;
    renderDetailHead();
    renderDetailPage();
  })
  .finally(() => {
    document.getElementById('loadingOverlay').classList.remove('active');
  });
}

function clearFilters() {
  document.getElementById('filterError').value = 'all';
  document.getElementById('filterLot').value = 'all';
  document.getElementById('dateFrom').value = '{{ date_min }}';
  document.getElementById('dateTo').value = '{{ date_max }}';
  applyFilters();
}

document.getElementById('mainForm').addEventListener('submit', () => {
  document.getElementById('loadingOverlay').classList.add('active');
});

document.addEventListener('DOMContentLoaded', initCharts);
document.addEventListener('DOMContentLoaded', () => { renderDetailHead(); renderDetailPage(); });
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
