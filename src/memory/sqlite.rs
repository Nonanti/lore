//! `SqliteStore`: persistent native memory engine (rusqlite, bundled).
//!
//! Records are stored as JSON in the `data` column; scope/tier/deleted have
//! separate columns (for SQL filtering). If an embedder is attached, embeddings
//! are computed at remember-time and recall becomes hybrid. Postgres/pgvector
//! is Alaz's domain; this module is dependency-free, embedded sqlite.

use super::embed::Embedder;
use super::retrieval;
use super::types::{ConsolidationReport, Memory, MemoryKind, Outcome, Query, Scope, Scored};
use super::MemoryStore;
use crate::error::{LoreError, Result};
use crate::id::MemoryId;
use async_trait::async_trait;
use chrono::Utc;
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};

fn sqlite_err(e: rusqlite::Error) -> LoreError {
    LoreError::Storage(format!("sqlite: {e}"))
}

/// Persistent, sqlite-backed memory store.
///
/// rusqlite is blocking: all trait methods offload work to a `spawn_blocking`
/// pool so that async runtime workers are not blocked by disk I/O.
/// Schema version — `meta.schema`. v2: `search_text` + `emb` columns (split
/// from JSON) + FTS5 index. Old files are automatically migrated at open.
const SCHEMA_VERSION: &str = "3";

pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    embedder: Option<Arc<dyn Embedder>>,
    reranker: Option<Arc<dyn super::rerank::Reranker>>,
    /// File path (None for in-memory) — allows consolidation to open a separate connection.
    path: Option<String>,
}

impl SqliteStore {
    /// Opens a file-backed store (creates if absent).
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path).map_err(sqlite_err)?;
        Self::init(conn, Some(path.to_string()))
    }

    /// In-memory sqlite (for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(sqlite_err)?;
        Self::init(conn, None)
    }

    fn init(mut conn: Connection, path: Option<String>) -> Result<Self> {
        // WAL: concurrent reader/writer (CLI ↔ service share the same DB);
        // busy_timeout: wait 5s on locked DB instead of failing immediately.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(sqlite_err)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                scope_agent TEXT,
                is_world    INTEGER NOT NULL,
                tier        TEXT NOT NULL,
                deleted     INTEGER NOT NULL,
                data        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scope ON memories(is_world, scope_agent);
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )
        .map_err(sqlite_err)?;
        Self::migrate(&mut conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: None,
            reranker: None,
            path,
        })
    }

    /// Attaches a reranker (builder): used when `Query.rerank` is set
    /// (default: the native lexical reranker). Runs inside the blocking
    /// pool — the right place for cross-encoder CPU work.
    pub fn with_reranker(mut self, reranker: Arc<dyn super::rerank::Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Migrates the schema to the current version (idempotent; single transaction).
    /// v1→v2: adds `search_text` (normalized token sequence) and `emb` (f32 LE
    /// BLOB) columns, extracts embedding from JSON, creates FTS5 index +
    /// sync triggers, and rebuilds the index.
    fn migrate(conn: &mut Connection) -> Result<()> {
        let has_col = |conn: &Connection, name: &str| -> Result<bool> {
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = ?1",
                    params![name],
                    |r| r.get(0),
                )
                .map_err(sqlite_err)?;
            Ok(n > 0)
        };

        let ver = Self::meta_get(conn, "schema")?.unwrap_or_else(|| "1".into());
        if ver == SCHEMA_VERSION {
            return Ok(());
        }

        // BEGIN IMMEDIATE: two tasks on the same agent can open the store
        // concurrently (parallel daemon workers); a deferred transaction
        // would fail outright with SQLITE_BUSY_SNAPSHOT on the read→write
        // upgrade (busy_timeout cannot help). IMMEDIATE acquires the write
        // lock upfront and waits via busy_timeout instead.
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(sqlite_err)?;
        // v1-only work (column adds, row re-encode backfill, FTS build) is
        // gated on the RECORDED version: a v2→v3 upgrade must not rewrite
        // every row + rebuild FTS for nothing (review #6). Fresh DBs report
        // "1" (empty meta) and take this path over empty tables — cheap.
        let from_v1 = ver == "1";
        if from_v1 {
            if !has_col(&tx, "search_text")? {
                tx.execute_batch(
                    "ALTER TABLE memories ADD COLUMN search_text TEXT NOT NULL DEFAULT ''",
                )
                .map_err(sqlite_err)?;
            }
            if !has_col(&tx, "emb")? {
                tx.execute_batch("ALTER TABLE memories ADD COLUMN emb BLOB")
                    .map_err(sqlite_err)?;
            }

            // Backfill: v1 rows have the embedding inside JSON — extract it into
            // BLOB, generate search_text. (decode_row also reads v1 JSON: if the
            // emb column is empty, the JSON embedding is used.) Rows are consumed
            // LAZILY — the entire table is not loaded into RAM (no single-pass
            // memory spike on large installations).
            {
                let mut stmt = tx
                    .prepare("SELECT id, data FROM memories")
                    .map_err(sqlite_err)?;
                let mut rows = stmt.query([]).map_err(sqlite_err)?;
                while let Some(row) = rows.next().map_err(sqlite_err)? {
                    let id: String = row.get(0).map_err(sqlite_err)?;
                    let data: String = row.get(1).map_err(sqlite_err)?;
                    // A single corrupt JSON row should not kill the entire
                    // migration — skip and warn, consistent with the persona
                    // policy where one bad record does not stop the service.
                    let mem: Memory = match serde_json::from_str(&data) {
                        Ok(m) => m,
                        Err(e) => {
                            tracing::warn!(id = %id, error = %e, "skipping corrupt row during v1→v2 migration");
                            continue;
                        }
                    };
                    let (slim, emb, stext) = Self::encode_row(&mem)?;
                    tx.execute(
                        "UPDATE memories SET data = ?1, search_text = ?2, emb = ?3 WHERE id = ?4",
                        params![slim, stext, emb, id],
                    )
                    .map_err(sqlite_err)?;
                }
            }

            // FTS5 (external content) + sync triggers. Note: upsert uses
            // `ON CONFLICT DO UPDATE` — `INSERT OR REPLACE` would skip the delete
            // trigger and leave the index stale.
            tx.execute_batch(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                search_text, content='memories', content_rowid='rowid', tokenize='unicode61');
             CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, search_text) VALUES (new.rowid, new.search_text);
             END;
             CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, search_text)
                    VALUES('delete', old.rowid, old.search_text);
             END;
             CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, search_text)
                    VALUES('delete', old.rowid, old.search_text);
                INSERT INTO memories_fts(rowid, search_text) VALUES (new.rowid, new.search_text);
             END;
             INSERT INTO memories_fts(memories_fts) VALUES('rebuild');",
            )
            .map_err(sqlite_err)?;
        } // from_v1

        // v3: entity inverted index feeding the recall graph leg. Backfilled
        // here (lazy row consumption, corrupt rows skipped — same policy as
        // the v1→v2 backfill above); maintained incrementally by `remember`.
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS entities (
                memory_id TEXT NOT NULL,
                entity    TEXT NOT NULL,
                PRIMARY KEY (memory_id, entity)
             ) WITHOUT ROWID;
             CREATE INDEX IF NOT EXISTS idx_entities_entity ON entities(entity);",
        )
        .map_err(sqlite_err)?;
        let ent_rows: i64 = tx
            .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
            .map_err(sqlite_err)?;
        if ent_rows == 0 {
            let mut stmt = tx
                .prepare("SELECT id, data FROM memories")
                .map_err(sqlite_err)?;
            let mut rows = stmt.query([]).map_err(sqlite_err)?;
            while let Some(row) = rows.next().map_err(sqlite_err)? {
                let id: String = row.get(0).map_err(sqlite_err)?;
                let data: String = row.get(1).map_err(sqlite_err)?;
                let mem: Memory = match serde_json::from_str(&data) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(id = %id, error = %e, "skipping corrupt row during entity backfill");
                        continue;
                    }
                };
                for ent in super::graph::extract_entities(&mem) {
                    tx.execute(
                        "INSERT OR IGNORE INTO entities (memory_id, entity) VALUES (?1, ?2)",
                        params![id, ent],
                    )
                    .map_err(sqlite_err)?;
                }
            }
        }

        Self::meta_set(&tx, "schema", SCHEMA_VERSION)?;
        tx.commit().map_err(sqlite_err)?;
        tracing::info!(schema = SCHEMA_VERSION, "sqlite schema migrated");
        Ok(())
    }

    /// f32 vector → little-endian BLOB.
    fn emb_to_blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// BLOB → f32 vector.
    fn blob_to_emb(b: &[u8]) -> Vec<f32> {
        if !b.len().is_multiple_of(4) {
            tracing::warn!(
                blob_len = b.len(),
                "embedding BLOB length is not a multiple of 4 — trailing bytes discarded"
            );
        }
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Splits a record into columns: `(slim_json, emb_blob, search_text)`.
    /// Embedding is extracted from JSON (no double storage); `search_text` is
    /// a normalized token sequence for FTS and pre-filtering — uses the same
    /// `tokenize` as scoring.
    fn encode_row(mem: &Memory) -> Result<(String, Option<Vec<u8>>, String)> {
        let mut slim = mem.clone();
        let emb = slim.embedding.take();
        let stext = retrieval::tokenize(&mem.searchable_text()).join(" ");
        Ok((
            serde_json::to_string(&slim)?,
            emb.map(|v| Self::emb_to_blob(&v)),
            stext,
        ))
    }

    /// Reconstructs a record from columns: emb BLOB takes precedence over any
    /// embedding in JSON (v1 legacy).
    fn decode_row(data: &str, emb: Option<Vec<u8>>) -> Result<Memory> {
        let mut m: Memory = serde_json::from_str(data)?;
        if let Some(b) = emb {
            m.embedding = Some(Self::blob_to_emb(&b));
        }
        Ok(m)
    }

    /// Runs a blocking sqlite operation on the tokio blocking pool.
    /// Closure receives `&mut Connection` — can open transactions.
    async fn blocking<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            // Recover from poisoned mutex: a panic in a previous critical
            // section leaves the data intact; abandoning the lock would make
            // the store permanently unusable.
            let mut conn = conn.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut conn)
        })
        .await
        .map_err(|e| LoreError::Storage(format!("blocking task: {e}")))?
    }

    /// Attaches an embedder (builder). Compares against the stored embedder
    /// signature: if different, warns (old vectors DO NOT MATCH in the new
    /// space — migrate with `lore reembed`); if no signature is stored,
    /// writes it.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        let sig = embedder.signature();
        {
            // Recover from poisoned mutex (see `blocking` for rationale).
            let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
            match Self::meta_get(&conn, "embedder") {
                Ok(Some(stored)) if stored != sig => {
                    tracing::warn!(
                        from = %stored,
                        to = %sig,
                        "embedder changed: old vectors do not match in new space, \
                         migrate with `lore reembed`"
                    );
                }
                Ok(None) => {
                    if let Err(e) = Self::meta_set(&conn, "embedder", &sig) {
                        tracing::warn!(error = %e, "embedder signature could not be written");
                    }
                }
                Ok(Some(_)) => {} // same signature — no problem
                Err(e) => tracing::warn!(error = %e, "embedder signature could not be read"),
            }
        }
        self.embedder = Some(embedder);
        self
    }

    fn meta_get(conn: &Connection, key: &str) -> Result<Option<String>> {
        use rusqlite::OptionalExtension;
        conn.query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
            r.get(0)
        })
        .optional()
        .map_err(sqlite_err)
    }

    fn meta_set(conn: &Connection, key: &str, value: &str) -> Result<()> {
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2)\n             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )
        .map_err(sqlite_err)?;
        Ok(())
    }

    /// Recomputes embeddings for all live records using the active embedder
    /// and updates the signature — required when switching embedders
    /// (hashing ↔ neural). Returns the number of re-embedded records.
    pub async fn reembed(&self) -> Result<usize> {
        const REEMBED_CHUNK: usize = 500;
        let embedder = self
            .embedder
            .clone()
            .ok_or_else(|| LoreError::InvalidInput("no embedder attached".into()))?;
        self.blocking(move |conn| Self::reembed_chunked(conn, &embedder, REEMBED_CHUNK))
            .await
    }

    /// Re-embedding worker: processes records in `chunk`-sized transactions.
    /// Partial commits instead of one giant transaction — avoids long write
    /// locks + recall-touch blocking on large stores. Atomicity: the signature
    /// (`meta`) is written LAST → an interrupted reembed will trigger a
    /// signature mismatch warning on the next open and can be re-run
    /// (self-healing).
    fn reembed_chunked(
        conn: &mut Connection,
        embedder: &Arc<dyn Embedder>,
        chunk: usize,
    ) -> Result<usize> {
        let live = Self::load_all_live(conn)?;
        let mut count = 0;
        for part in live.chunks(chunk.max(1)) {
            let tx = conn.transaction().map_err(sqlite_err)?;
            for mem in part {
                let mut mem = mem.clone();
                mem.embedding = Some(embedder.embed(&mem.searchable_text()));
                Self::upsert(&tx, &mem)?;
                count += 1;
            }
            tx.commit().map_err(sqlite_err)?;
        }
        Self::meta_set(conn, "embedder", &embedder.signature())?;
        Ok(count)
    }

    /// Total row count (including soft-deleted; for diagnostics/testing).
    pub fn total_rows(&self) -> Result<usize> {
        // Recover from poisoned mutex (see `blocking` for rationale).
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .map_err(sqlite_err)?;
        Ok(n as usize)
    }

    /// Writes/updates a record. `ON CONFLICT DO UPDATE` is required:
    /// `INSERT OR REPLACE` performs delete-then-reinsert — the rowid changes
    /// and the delete trigger is skipped (when recursive_triggers is off),
    /// leaving the FTS index stale.
    fn upsert(conn: &Connection, mem: &Memory) -> Result<()> {
        let (agent, is_world) = match &mem.scope {
            Scope::Agent(a) => (Some(a.to_string()), 0i64),
            Scope::World => (None, 1i64),
        };
        let tier = format!("{:?}", mem.tier());
        let (data, emb, stext) = Self::encode_row(mem)?;
        conn.execute(
            "INSERT INTO memories (id, scope_agent, is_world, tier, deleted, data, search_text, emb)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(id) DO UPDATE SET
                scope_agent = excluded.scope_agent,
                is_world    = excluded.is_world,
                tier        = excluded.tier,
                deleted     = excluded.deleted,
                data        = excluded.data,
                search_text = excluded.search_text,
                emb         = excluded.emb",
            params![
                mem.id.to_string(),
                agent,
                is_world,
                tier,
                mem.deleted_at.is_some() as i64,
                data,
                stext,
                emb
            ],
        )
        .map_err(sqlite_err)?;
        Ok(())
    }

    /// Loads records visible to a scope. When `include_deleted=false`,
    /// soft-deleted rows are filtered at the SQL level — avoiding the load +
    /// JSON parse cost. (Soft delete is permanent; without this filter, deleted
    /// rows would be carried on every recall forever.)
    fn load_by_scope(
        conn: &Connection,
        scope: &Scope,
        include_deleted: bool,
    ) -> Result<Vec<Memory>> {
        let mut out = Vec::new();
        let mut push_rows = |sql: &str, bind: Option<String>| -> Result<()> {
            let mut stmt = conn.prepare(sql).map_err(sqlite_err)?;
            let mapper =
                |r: &rusqlite::Row| Ok((r.get::<_, String>(0)?, r.get::<_, Option<Vec<u8>>>(1)?));
            let rows = match &bind {
                Some(b) => stmt.query_map(params![b], mapper).map_err(sqlite_err)?,
                None => stmt.query_map([], mapper).map_err(sqlite_err)?,
            };
            for row in rows {
                let (s, e) = row.map_err(sqlite_err)?;
                out.push(Self::decode_row(&s, e)?);
            }
            Ok(())
        };

        let del = if include_deleted {
            ""
        } else {
            " AND deleted = 0"
        };
        match scope {
            Scope::Agent(a) => push_rows(
                &format!(
                    "SELECT data, emb FROM memories WHERE (is_world = 1 OR scope_agent = ?1){del}"
                ),
                Some(a.to_string()),
            )?,
            Scope::World => push_rows(
                &format!("SELECT data, emb FROM memories WHERE is_world = 1{del}"),
                None,
            )?,
        }
        Ok(out)
    }

    fn load_all_live(conn: &Connection) -> Result<Vec<Memory>> {
        let mut stmt = conn
            .prepare("SELECT data, emb FROM memories WHERE deleted = 0")
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<Vec<u8>>>(1)?))
            })
            .map_err(sqlite_err)?;
        let mut out = Vec::new();
        for row in rows {
            let (s, e) = row.map_err(sqlite_err)?;
            out.push(Self::decode_row(&s, e)?);
        }
        Ok(out)
    }

    fn load_one(conn: &Connection, id: &MemoryId) -> Result<Option<Memory>> {
        let mut stmt = conn
            .prepare_cached("SELECT data, emb FROM memories WHERE id = ?1")
            .map_err(sqlite_err)?;
        let mut rows = stmt.query(params![id.to_string()]).map_err(sqlite_err)?;
        match rows.next().map_err(sqlite_err)? {
            Some(row) => {
                let s: String = row.get(0).map_err(sqlite_err)?;
                let e: Option<Vec<u8>> = row.get(1).map_err(sqlite_err)?;
                Ok(Some(Self::decode_row(&s, e)?))
            }
            None => Ok(None),
        }
    }

    /// Loads the given set of ids with a parameterized `IN (...)` batch
    /// (chunked at 500 ids). Result ordering matches the input `ids` order.
    /// Duplicate ids are silently deduplicated — only the first occurrence
    /// is preserved in the output.
    fn load_by_ids(conn: &Connection, ids: &[String]) -> Result<Vec<Memory>> {
        debug_assert!(
            ids.len() == ids.iter().collect::<std::collections::HashSet<_>>().len(),
            "load_by_ids called with duplicate ids — they will be silently deduplicated"
        );
        const CHUNK: usize = 500;
        let mut by_id: std::collections::HashMap<String, Memory> =
            std::collections::HashMap::with_capacity(ids.len());
        for chunk in ids.chunks(CHUNK) {
            let placeholders: String = (0..chunk.len())
                .map(|i| format!("?{}", i + 1))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!("SELECT id, data, emb FROM memories WHERE id IN ({placeholders})");
            let mut stmt = conn.prepare(&sql).map_err(sqlite_err)?;
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            let rows = stmt
                .query_map(params.as_slice(), |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<Vec<u8>>>(2)?,
                    ))
                })
                .map_err(sqlite_err)?;
            for row in rows {
                let (id, data, emb) = row.map_err(sqlite_err)?;
                by_id.insert(id, Self::decode_row(&data, emb)?);
            }
        }
        // Preserve input ordering.
        let out: Vec<Memory> = ids.iter().filter_map(|id| by_id.remove(id)).collect();
        Ok(out)
    }

    /// Loads keyword candidates via FTS in a single query: rows containing
    /// any of the query tokens (OR) are returned directly as `data, emb` —
    /// no second lookup per candidate. Because `search_text` is
    /// pre-normalized, the match is identical to scoring's `kw > 0` predicate.
    fn fts_load_candidates(
        conn: &Connection,
        scope: &Scope,
        q_terms: &[String],
        include_deleted: bool,
    ) -> Result<Vec<Memory>> {
        let expr = q_terms
            .iter()
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(" OR ");
        let del = if include_deleted {
            ""
        } else {
            " AND m.deleted = 0"
        };
        let mut out = Vec::new();
        let mut run = |sql: &str, bind: Option<String>| -> Result<()> {
            let mut stmt = conn.prepare(sql).map_err(sqlite_err)?;
            let mapper =
                |r: &rusqlite::Row| Ok((r.get::<_, String>(0)?, r.get::<_, Option<Vec<u8>>>(1)?));
            let rows = match &bind {
                Some(b) => stmt
                    .query_map(params![expr, b], mapper)
                    .map_err(sqlite_err)?,
                None => stmt.query_map(params![expr], mapper).map_err(sqlite_err)?,
            };
            for row in rows {
                let (s, e) = row.map_err(sqlite_err)?;
                out.push(Self::decode_row(&s, e)?);
            }
            Ok(())
        };
        // `IN (SELECT rowid ...)` rationale: a plain JOIN caused the planner
        // to pick idx_scope as the outer loop and probe FTS for every row
        // (~190ms on 10k rows); the subquery runs FTS once and looks up by
        // rowid set (~1ms).
        match scope {
            Scope::Agent(a) => run(
                &format!(
                    "SELECT m.data, m.emb FROM memories m
                     WHERE m.rowid IN (SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?1)
                       AND (m.is_world = 1 OR m.scope_agent = ?2){del}"
                ),
                Some(a.to_string()),
            )?,
            Scope::World => run(
                &format!(
                    "SELECT m.data, m.emb FROM memories m
                     WHERE m.rowid IN (SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?1)
                       AND m.is_world = 1{del}"
                ),
                None,
            )?,
        }
        Ok(out)
    }

    /// Lightweight scan for semantic queries: reads `(id, search_text, emb)` —
    /// NO JSON parse. Candidate: rows that have a keyword hit OR pass the
    /// semantic gate/token fallback (synchronized with scoring's candidate
    /// predicate — see [`retrieval::semantic_prefilter_hit`]).
    fn semantic_candidate_ids(
        conn: &Connection,
        scope: &Scope,
        q_terms: &[String],
        q_emb: Option<&[f32]>,
        embedder: Option<&dyn Embedder>,
        include_deleted: bool,
    ) -> Result<Vec<String>> {
        let del = if include_deleted {
            ""
        } else {
            " AND deleted = 0"
        };
        let mut rows_out: Vec<(String, String, Option<Vec<u8>>)> = Vec::new();
        let mut run = |sql: &str, bind: Option<String>| -> Result<()> {
            let mut stmt = conn.prepare(sql).map_err(sqlite_err)?;
            let mapper = |r: &rusqlite::Row| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<Vec<u8>>>(2)?,
                ))
            };
            let rows = match &bind {
                Some(b) => stmt.query_map(params![b], mapper).map_err(sqlite_err)?,
                None => stmt.query_map([], mapper).map_err(sqlite_err)?,
            };
            for row in rows {
                rows_out.push(row.map_err(sqlite_err)?);
            }
            Ok(())
        };
        match scope {
            Scope::Agent(a) => run(
                &format!(
                    "SELECT id, search_text, emb FROM memories
                     WHERE (is_world = 1 OR scope_agent = ?1){del}"
                ),
                Some(a.to_string()),
            )?,
            Scope::World => run(
                &format!("SELECT id, search_text, emb FROM memories WHERE is_world = 1{del}"),
                None,
            )?,
        }

        let mut cache: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();
        let mut out = Vec::new();
        for (id, stext, emb_blob) in rows_out {
            let kw_hit = {
                let doc: std::collections::HashSet<&str> = stext.split(' ').collect();
                q_terms.iter().any(|t| doc.contains(t.as_str()))
            };
            let hit = kw_hit || {
                let emb = emb_blob.map(|b| Self::blob_to_emb(&b));
                retrieval::semantic_prefilter_hit(
                    q_emb,
                    emb.as_deref(),
                    &stext,
                    q_terms.len(),
                    embedder,
                    &mut cache,
                )
            };
            if hit {
                out.push(id);
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl MemoryStore for SqliteStore {
    async fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let id = id.clone();
        self.blocking(move |conn| Self::load_one(conn, &id)).await
    }

    async fn remember(&self, mut mem: Memory) -> Result<MemoryId> {
        let embedder = self.embedder.clone();
        self.blocking(move |conn| {
            // Embedding is also CPU work (especially neural) — stays in the blocking pool.
            if let Some(e) = &embedder {
                if mem.embedding.is_none() {
                    mem.embedding = Some(e.embed(&mem.searchable_text()));
                }
            }
            let id = mem.id.clone();
            // One transaction for row + entity refresh: a crash between the
            // DELETE and the INSERTs would otherwise sever this record's
            // entity links PERMANENTLY (schema is already v3, so the
            // backfill never re-runs). Review finding #1.
            // IMMEDIATE: parallel daemon workers write to the same agent DB;
            // a deferred read→write upgrade fails with SQLITE_BUSY(_SNAPSHOT)
            // instead of waiting on busy_timeout — same lesson as `migrate`.
            let tx = conn
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .map_err(sqlite_err)?;
            Self::upsert(&tx, &mem)?;
            // Refresh the entity index for this record (remember may overwrite
            // an existing id). Reinforce paths skip this — text never changes.
            tx.execute(
                "DELETE FROM entities WHERE memory_id = ?1",
                params![id.to_string()],
            )
            .map_err(sqlite_err)?;
            for ent in super::graph::extract_entities(&mem) {
                tx.execute(
                    "INSERT OR IGNORE INTO entities (memory_id, entity) VALUES (?1, ?2)",
                    params![id.to_string(), ent],
                )
                .map_err(sqlite_err)?;
            }
            tx.commit().map_err(sqlite_err)?;
            Ok(id)
        })
        .await
    }

    /// Recall runs three paths — all funnel into the same scoring core (parity
    /// test: `sqlite_recall_matches_in_memory_reference`):
    ///
    /// 1. **Browse** (textless): entire scope is loaded (legacy behavior).
    /// 2. **Keyword** (semantic off): FTS5 index → only matching rows are
    ///    loaded; no full table scan.
    /// 3. **Semantic**: lightweight scan `(id, search_text, emb)` —
    ///    pre-filter without JSON parse cost; only candidates are fully loaded.
    async fn recall(&self, scope: &Scope, query: &Query) -> Result<Vec<Scored<Memory>>> {
        let embedder = self.embedder.clone();
        let reranker = self.reranker.clone();
        let scope = scope.clone();
        let query = query.clone();
        self.blocking(move |conn| {
            let now = Utc::now();
            let has_text = !query.text.trim().is_empty();
            let embed_src = query.embed_text.as_deref().unwrap_or(&query.text);
            let q_emb = match (&embedder, has_text) {
                (Some(e), true) => Some(e.embed(embed_src)),
                _ => None,
            };
            let q_terms = retrieval::tokenize(&query.text);

            let candidates: Vec<Memory> = if !has_text || q_terms.is_empty() {
                Self::load_by_scope(conn, &scope, query.include_deleted)?
            } else if !query.semantic {
                Self::fts_load_candidates(conn, &scope, &q_terms, query.include_deleted)?
            } else {
                let ids = Self::semantic_candidate_ids(
                    conn,
                    &scope,
                    &q_terms,
                    q_emb.as_deref(),
                    embedder.as_deref(),
                    query.include_deleted,
                )?;
                Self::load_by_ids(conn, &ids)?
            };

            // Single Scorer across the scan: token embedding cache is shared.
            let mut scorer = retrieval::Scorer::new(embedder.as_deref());
            let mut scored: Vec<Scored<Memory>> = Vec::new();
            for mem in candidates {
                if mem.deleted_at.is_some() && !query.include_deleted {
                    continue;
                }
                if let Some(tiers) = &query.tiers {
                    if !tiers.contains(&mem.tier()) {
                        continue;
                    }
                }
                if let Some(min_imp) = query.min_importance {
                    if mem.importance < min_imp {
                        continue;
                    }
                }
                let (score, signals) = scorer.score(&mem, &query, q_emb.as_deref(), now);
                if has_text && score <= 0.0 {
                    continue;
                }
                scored.push(Scored {
                    item: mem,
                    score,
                    signals,
                });
            }

            // Graph expansion leg: top seeds pull 1-hop entity neighbors in
            // with a damped score — multi-hop answers no single record
            // matches. Mirrors the InMemoryStore implementation.
            if query.graph && has_text && !scored.is_empty() {
                scored.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let seen: std::collections::HashSet<String> =
                    scored.iter().map(|s| s.item.id.to_string()).collect();
                let mut seed_entities: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for s in scored.iter().take(retrieval::GRAPH_SEED_K) {
                    seed_entities.extend(super::graph::extract_entities(&s.item));
                }
                if !seed_entities.is_empty() {
                    // Entities of ≤3 records (~30–60 terms) stay far below
                    // SQLITE_MAX_VARIABLE_NUMBER (999 default; 32766 on
                    // ≥3.32); truncate defensively regardless. Per-recall
                    // statement compilation is an accepted tradeoff at this
                    // size — revisit with a temp-table join if graph recall
                    // ever shows up in profiles.
                    let mut ents: Vec<String> = seed_entities.into_iter().collect();
                    ents.sort();
                    ents.truncate(400);
                    let placeholders: String = (0..ents.len())
                        .map(|i| format!("?{}", i + 1))
                        .collect::<Vec<_>>()
                        .join(",");
                    let sql = format!(
                        "SELECT memory_id, COUNT(*) AS c FROM entities
                         WHERE entity IN ({placeholders})
                         GROUP BY memory_id ORDER BY c DESC, memory_id ASC"
                    );
                    let mut stmt = conn.prepare(&sql).map_err(sqlite_err)?;
                    let sql_params: Vec<&dyn rusqlite::types::ToSql> = ents
                        .iter()
                        .map(|e| e as &dyn rusqlite::types::ToSql)
                        .collect();
                    let mut rows = stmt.query(&sql_params[..]).map_err(sqlite_err)?;
                    let mut nb_ids: Vec<String> = Vec::new();
                    while let Some(row) = rows.next().map_err(sqlite_err)? {
                        let nid: String = row.get(0).map_err(sqlite_err)?;
                        if !seen.contains(&nid) {
                            nb_ids.push(nid);
                            if nb_ids.len() >= retrieval::GRAPH_NEIGHBOR_CAP {
                                break;
                            }
                        }
                    }
                    let best_seed = scored.first().map(|s| s.score).unwrap_or(0.0);
                    let mut neighbors: Vec<Memory> = Vec::new();
                    for mem in Self::load_by_ids(conn, &nb_ids)? {
                        if !scope.sees(&mem.scope)
                            || (mem.deleted_at.is_some() && !query.include_deleted)
                        {
                            continue;
                        }
                        if let Some(tiers) = &query.tiers {
                            if !tiers.contains(&mem.tier()) {
                                continue;
                            }
                        }
                        if let Some(min_imp) = query.min_importance {
                            if mem.importance < min_imp {
                                continue;
                            }
                        }
                        neighbors.push(mem);
                    }
                    retrieval::append_graph_neighbors(&mut scored, neighbors, best_seed);
                }
            }

            Ok(retrieval::finalize(scored, &query, reranker.as_deref()))
        })
        .await
    }

    async fn reinforce(&self, id: &MemoryId, outcome: Outcome) -> Result<()> {
        let id = id.clone();
        self.blocking(move |conn| {
            let mut mem =
                Self::load_one(conn, &id)?.ok_or_else(|| LoreError::NotFound(id.to_string()))?;
            mem.last_access = Utc::now();
            mem.access_count = mem.access_count.saturating_add(1);
            match outcome {
                Outcome::Accessed => {}
                Outcome::Success => {
                    if let MemoryKind::Procedural { successes, .. } = &mut mem.kind {
                        *successes = successes.saturating_add(1);
                    }
                }
                Outcome::Failure => {
                    if let MemoryKind::Procedural { failures, .. } = &mut mem.kind {
                        *failures = failures.saturating_add(1);
                    }
                }
            }
            Self::upsert(conn, &mem)?;
            Ok(())
        })
        .await
    }

    /// Batch reinforcement: sequential read-modify-write under a single
    /// blocking call, single lock. Missing ids are skipped (does not kill the
    /// batch). Accessed outcome increments counter/time with the same logic as
    /// `reinforce`.
    async fn reinforce_many(&self, ids: &[MemoryId], outcome: Outcome) -> Result<()> {
        let ids = ids.to_vec();
        self.blocking(move |conn| {
            let tx = conn.transaction().map_err(sqlite_err)?;
            let now = Utc::now();
            for id in &ids {
                let Some(mut mem) = Self::load_one(&tx, id)? else {
                    continue; // skip — batch processes the rest
                };
                mem.last_access = now;
                mem.access_count = mem.access_count.saturating_add(1);
                match outcome {
                    Outcome::Accessed => {}
                    Outcome::Success => {
                        if let MemoryKind::Procedural { successes, .. } = &mut mem.kind {
                            *successes = successes.saturating_add(1);
                        }
                    }
                    Outcome::Failure => {
                        if let MemoryKind::Procedural { failures, .. } = &mut mem.kind {
                            *failures = failures.saturating_add(1);
                        }
                    }
                }
                Self::upsert(&tx, &mem)?;
            }
            tx.commit().map_err(sqlite_err)?;
            Ok(())
        })
        .await
    }

    async fn forget(&self, id: &MemoryId) -> Result<()> {
        let id = id.clone();
        self.blocking(move |conn| {
            let mut mem =
                Self::load_one(conn, &id)?.ok_or_else(|| LoreError::NotFound(id.to_string()))?;
            mem.deleted_at = Some(Utc::now());
            Self::upsert(conn, &mem)?;
            Ok(())
        })
        .await
    }

    async fn count(&self, scope: &Scope) -> Result<usize> {
        let scope = scope.clone();
        self.blocking(move |conn| {
            let n: i64 = match &scope {
                Scope::World => conn
                    .query_row(
                        "SELECT COUNT(*) FROM memories WHERE deleted = 0 AND is_world = 1",
                        [],
                        |r| r.get(0),
                    )
                    .map_err(sqlite_err)?,
                Scope::Agent(a) => conn
                    .query_row(
                        "SELECT COUNT(*) FROM memories
                         WHERE deleted = 0 AND (is_world = 1 OR scope_agent = ?1)",
                        params![a.to_string()],
                        |r| r.get(0),
                    )
                    .map_err(sqlite_err)?,
            };
            Ok(n as usize)
        })
        .await
    }

    async fn export(&self) -> Result<Vec<Memory>> {
        self.blocking(|conn| Self::load_all_live(conn)).await
    }

    /// File-backed stores run consolidation on a SEPARATE connection: the O(n²)
    /// dedup scan does not hold the main connection mutex — remember/recall
    /// are not blocked during this time (WAL allows concurrent readers and
    /// writers; write conflicts are resolved via busy_timeout). In-memory DBs
    /// cannot open a second connection → falls back to the shared path.
    async fn consolidate(&self) -> Result<ConsolidationReport> {
        match self.path.clone() {
            Some(path) => tokio::task::spawn_blocking(move || {
                let mut conn = Connection::open(&path).map_err(sqlite_err)?;
                // WAL has a single writer: recall-touch and reflect write
                // heavily from the main connection while consolidation waits.
                // 5s may not suffice under heavy burst (observed in soak) —
                // 30s: consolidation is not urgent, it can wait.
                conn.execute_batch("PRAGMA busy_timeout=30000;")
                    .map_err(sqlite_err)?;
                Self::consolidate_on(&mut conn)
            })
            .await
            .map_err(|e| LoreError::Storage(format!("blocking task: {e}")))?,
            None => self.blocking(Self::consolidate_on).await,
        }
    }
}

impl SqliteStore {
    /// Connection-independent core of consolidation.
    /// Forget writes are in a SINGLE transaction: one commit instead of N autocommits.
    /// BEGIN IMMEDIATE: write lock is acquired upfront (waits via
    /// busy_timeout) — with deferred mode, the transition from read to write
    /// would immediately fail with SQLITE_BUSY_SNAPSHOT if the main connection
    /// wrote in between (observed in soak).
    fn consolidate_on(conn: &mut Connection) -> Result<ConsolidationReport> {
        let now = Utc::now();
        let policy = super::evolution::ForgetPolicy::default();
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(sqlite_err)?;
        let live = Self::load_all_live(&tx)?;
        let scanned = live.len();
        let p = super::evolution::plan(&live, &policy, now);
        for id in &p.to_forget {
            if let Some(mut m) = Self::load_one(&tx, id)? {
                m.deleted_at = Some(now);
                Self::upsert(&tx, &m)?;
            }
        }
        // Index hygiene (review #7): soft-deleted records are filtered at
        // read, but their entity rows would otherwise accumulate forever.
        tx.execute(
            "DELETE FROM entities WHERE memory_id IN
                (SELECT id FROM memories WHERE deleted = 1)",
            [],
        )
        .map_err(sqlite_err)?;
        tx.commit().map_err(sqlite_err)?;
        Ok(ConsolidationReport {
            scanned,
            merged: p.merged,
            forgotten: p.forgotten,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::AgentId;
    use crate::memory::embed::HashingEmbedder;
    use crate::memory::types::SemanticCat;

    fn scope() -> Scope {
        Scope::Agent(AgentId::new())
    }

    /// Temporary file DB path (cleaned up with WAL side-files).
    struct TmpDb(String);
    impl TmpDb {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!("lore-test-{}.db", ulid::Ulid::new()));
            Self(p.to_string_lossy().into_owned())
        }
    }
    impl Drop for TmpDb {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{}", self.0, suffix));
            }
        }
    }

    #[tokio::test]
    async fn schema_v2_migrates_v1_file_and_keeps_data() {
        // v1 file (no search_text/emb columns or FTS; embedding inside JSON)
        // should be migrated to v2 on open: data preserved, FTS created,
        // emb split to BLOB.
        let tmp = TmpDb::new();
        let e = HashingEmbedder::new();
        let s = scope();
        {
            let conn = Connection::open(&tmp.0).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id TEXT PRIMARY KEY, scope_agent TEXT, is_world INTEGER NOT NULL,
                    tier TEXT NOT NULL, deleted INTEGER NOT NULL, data TEXT NOT NULL);
                 CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
            )
            .unwrap();
            let mut m = Memory::semantic(s.clone(), "Rust is fast", SemanticCat::Fact);
            m.embedding = Some(e.embed(&m.searchable_text()));
            let agent = match &m.scope {
                Scope::Agent(a) => Some(a.to_string()),
                Scope::World => None,
            };
            conn.execute(
                "INSERT INTO memories (id, scope_agent, is_world, tier, deleted, data)
                 VALUES (?1, ?2, 0, 'Semantic', 0, ?3)",
                params![m.id.to_string(), agent, serde_json::to_string(&m).unwrap()],
            )
            .unwrap();
        }

        let store = SqliteStore::open(&tmp.0)
            .unwrap()
            .with_embedder(Arc::new(HashingEmbedder::new()));
        // Keyword recall (FTS path) finds the old record.
        let hits = store.recall(&s, &Query::new("rust")).await.unwrap();
        assert_eq!(hits.len(), 1, "v1 record found after migration");
        // Embedding is restored from BLOB (export path).
        let all = store.export().await.unwrap();
        assert!(
            all[0].embedding.is_some(),
            "embedding must not be lost during migration"
        );
        // v2 structures created: FTS table + schema stamp.
        {
            let conn = store.conn.lock().unwrap();
            let fts: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = 'memories_fts'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(fts, 1, "FTS table is created by migration");
        }
        assert_eq!(
            SqliteStore::meta_get(&store.conn.lock().unwrap(), "schema")
                .unwrap()
                .as_deref(),
            Some(SCHEMA_VERSION),
            "schema version stamped"
        );
    }

    #[tokio::test]
    async fn sqlite_recall_matches_in_memory_reference() {
        // FTS/lightweight pre-filtering must produce the SAME result set as
        // the reference full-scan (InMemoryStore) — keyword, semantic
        // morphology, and browse.
        use crate::memory::in_memory::InMemoryStore;
        let sq = SqliteStore::in_memory()
            .unwrap()
            .with_embedder(Arc::new(HashingEmbedder::new()));
        let im = InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new()));
        let s = scope();
        let docs = [
            ("Learned Rust", "learned ownership and borrow checker"),
            ("Tried Go", "goroutine channels interesting"),
            ("Saw a cat", "neighbor's cat came"),
            ("Sqlite note", "fts5 provides fast search"),
            ("Empty day", "walk by the sea"),
        ];
        for (t, b) in docs {
            // SAME record (including id/created_at) is written to both stores —
            // identities must be shared for parity comparison.
            let m = Memory::episodic(s.clone(), t, b);
            sq.remember(m.clone()).await.unwrap();
            im.remember(m).await.unwrap();
        }
        for q in [
            Query::new("rust ownership"),
            Query::new("learning").semantic(), // short query → token fallback
            Query::new("cat"),
            Query::new("").limit(10), // browse
        ] {
            let a = sq.recall(&s, &q).await.unwrap();
            let b = im.recall(&s, &q).await.unwrap();
            let mut ids_a: Vec<String> = a.iter().map(|x| x.item.id.to_string()).collect();
            let mut ids_b: Vec<String> = b.iter().map(|x| x.item.id.to_string()).collect();
            ids_a.sort();
            ids_b.sort();
            assert_eq!(ids_a, ids_b, "result sets equal (q={:?})", q.text);
            // Scores match per-id (same scoring core).
            for x in &a {
                let y = b
                    .iter()
                    .find(|y| y.item.id == x.item.id)
                    .expect("same record");
                assert!(
                    (x.score - y.score).abs() < 1e-5,
                    "score parity (q={:?}): {} vs {}",
                    q.text,
                    x.score,
                    y.score
                );
            }
        }
    }

    #[tokio::test]
    async fn consolidate_runs_on_separate_connection_for_file_db() {
        // File-backed store: consolidation runs on a separate connection (does
        // not hold the main mutex) and produces a report.
        let tmp = TmpDb::new();
        let store = SqliteStore::open(&tmp.0)
            .unwrap()
            .with_embedder(Arc::new(HashingEmbedder::new()));
        let s = scope();
        // Old auto-record eligible for forgetting.
        let mut old = Memory::episodic(s.clone(), "old exchange", "response")
            .with_importance(Memory::AUTO_IMPORTANCE);
        let past = Utc::now() - chrono::Duration::days(120);
        old.created_at = past;
        old.last_access = past;
        store.remember(old).await.unwrap();
        let report = store.consolidate().await.unwrap();
        assert_eq!(report.scanned, 1);
        assert_eq!(report.forgotten, 1, "old auto-record is forgotten");
    }

    #[tokio::test]
    async fn export_returns_only_live_records() {
        let store = SqliteStore::in_memory().unwrap();
        let s = scope();
        let keep = store
            .remember(Memory::semantic(s.clone(), "persistent", SemanticCat::Fact))
            .await
            .unwrap();
        let gone = store
            .remember(Memory::semantic(
                s.clone(),
                "to be deleted",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        store.forget(&gone).await.unwrap();

        let dump = store.export().await.unwrap();
        assert_eq!(dump.len(), 1, "only live records are dumped");
        assert_eq!(dump[0].id, keep);

        // Round-trip: dump can be imported into another store (id preserved).
        let other = SqliteStore::in_memory().unwrap();
        for m in dump {
            other.remember(m).await.unwrap();
        }
        let res = other.recall(&s, &Query::new("persistent")).await.unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].item.id, keep);
    }

    #[tokio::test]
    async fn reembed_migrates_to_new_embedder_space() {
        // Old space: 512 dimensions. Record is embedded in this space.
        let store = SqliteStore::in_memory()
            .unwrap()
            .with_embedder(Arc::new(HashingEmbedder::new()));
        let s = scope();
        store
            .remember(Memory::semantic(
                s.clone(),
                "user likes math",
                SemanticCat::Preference,
            ))
            .await
            .unwrap();

        // New space: 64 dimensions (different signature) — old vectors are now meaningless.
        let store = store.with_embedder(Arc::new(HashingEmbedder::with_params(64, 3)));
        let n = store.reembed().await.unwrap();
        assert_eq!(n, 1, "live record re-embedded");

        // Record is in the new dimension and semantic recall works in the new space.
        let res = store
            .recall(&s, &Query::new("math").semantic())
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].item.embedding.as_ref().unwrap().len(), 64);
    }

    #[tokio::test]
    async fn persists_and_recalls() {
        let store = SqliteStore::in_memory().unwrap();
        let s = scope();
        store
            .remember(Memory::semantic(
                s.clone(),
                "Rust is memory safe",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();

        let res = store.recall(&s, &Query::new("rust")).await.unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(store.total_rows().unwrap(), 1);
        assert_eq!(store.count(&s).await.unwrap(), 1);
        assert_eq!(store.count(&Scope::World).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn survives_reopen_on_disk() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lore-test-{}.db", AgentId::new()));
        let path_str = path.to_str().unwrap().to_string();
        let s = scope();

        {
            let store = SqliteStore::open(&path_str).unwrap();
            store
                .remember(Memory::episodic(
                    s.clone(),
                    "persistent event",
                    "will persist after restart",
                ))
                .await
                .unwrap();
        }
        // Reopen → record should still be there.
        {
            let store = SqliteStore::open(&path_str).unwrap();
            assert_eq!(store.total_rows().unwrap(), 1);
            let res = store.recall(&s, &Query::new("persistent")).await.unwrap();
            assert_eq!(res.len(), 1);
        }
        let _ = std::fs::remove_file(&path_str);
    }

    #[tokio::test]
    async fn scope_isolation_in_sqlite() {
        let store = SqliteStore::in_memory().unwrap();
        let a = scope();
        let b = scope();
        store
            .remember(Memory::semantic(
                a.clone(),
                "agent a secret alpha",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        store
            .remember(Memory::semantic(
                Scope::World,
                "world note alpha",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();

        assert_eq!(
            store.recall(&b, &Query::new("alpha")).await.unwrap().len(),
            1
        );
        assert_eq!(
            store.recall(&a, &Query::new("alpha")).await.unwrap().len(),
            2
        );
    }

    #[tokio::test]
    async fn semantic_recall_matches_morphological_variant() {
        let store = SqliteStore::in_memory()
            .unwrap()
            .with_embedder(Arc::new(HashingEmbedder::new()));
        let s = scope();
        store
            .remember(Memory::semantic(
                s.clone(),
                "User really likes mathematics",
                SemanticCat::Preference,
            ))
            .await
            .unwrap();

        // Plain keyword: "maths" != "mathematics" → no match.
        let plain = store.recall(&s, &Query::new("maths")).await.unwrap();
        assert_eq!(plain.len(), 0, "keyword misses morphology");

        // Semantic on: cosine catches it.
        let sem = store
            .recall(&s, &Query::new("math").semantic())
            .await
            .unwrap();
        assert_eq!(sem.len(), 1, "semantic catches morphology");
    }

    #[tokio::test]
    async fn soft_deleted_rows_filtered_in_sql_but_reachable_with_flag() {
        // Deleted rows are filtered at the SQL level in normal recall (no load
        // cost); still accessible with `with_deleted` (auditability).
        let store = SqliteStore::in_memory().unwrap();
        let s = scope();
        let id = store
            .remember(Memory::semantic(
                s.clone(),
                "secret note",
                SemanticCat::Fact,
            ))
            .await
            .unwrap();
        store.forget(&id).await.unwrap();

        let plain = store.recall(&s, &Query::new("secret")).await.unwrap();
        assert_eq!(plain.len(), 0, "deleted record invisible in normal recall");

        let with_del = store
            .recall(&s, &Query::new("secret").with_deleted())
            .await
            .unwrap();
        assert_eq!(with_del.len(), 1, "with_deleted sees deleted");
        assert_eq!(with_del[0].item.id, id);
    }

    #[tokio::test]
    async fn load_by_ids_batch_correctness_and_ordering() {
        // Fix 2: batch IN (...) returns results in input order and handles
        // missing ids gracefully.
        let store = SqliteStore::in_memory().unwrap();
        let s = scope();
        let a = store
            .remember(Memory::semantic(s.clone(), "alpha", SemanticCat::Fact))
            .await
            .unwrap();
        let b = store
            .remember(Memory::semantic(s.clone(), "beta", SemanticCat::Fact))
            .await
            .unwrap();
        let c = store
            .remember(Memory::semantic(s.clone(), "gamma", SemanticCat::Fact))
            .await
            .unwrap();
        // Request in reverse order + a missing id.
        let ids: Vec<String> = vec![
            c.to_string(),
            MemoryId::new().to_string(), // missing
            a.to_string(),
            b.to_string(),
        ];
        let conn = store.conn.lock().unwrap();
        let loaded = SqliteStore::load_by_ids(&conn, &ids).unwrap();
        assert_eq!(loaded.len(), 3, "missing id is skipped");
        assert_eq!(loaded[0].id, c, "order: c first");
        assert_eq!(loaded[1].id, a, "order: a second");
        assert_eq!(loaded[2].id, b, "order: b third");
    }

    #[tokio::test]
    async fn get_fetches_by_id_and_reinforce_many_batches() {
        // get: single record by id (None if absent).
        let store = SqliteStore::in_memory().unwrap();
        let s = scope();
        let a = store
            .remember(Memory::semantic(s.clone(), "alpha note", SemanticCat::Fact))
            .await
            .unwrap();
        let b = store
            .remember(Memory::semantic(s.clone(), "beta note", SemanticCat::Fact))
            .await
            .unwrap();
        assert!(store.get(&a).await.unwrap().is_some());
        assert!(store.get(&MemoryId::new()).await.unwrap().is_none());

        // reinforce_many: batch reinforcement in a single call; missing ids
        // are skipped (does not kill the batch).
        store
            .reinforce_many(&[a.clone(), b.clone(), MemoryId::new()], Outcome::Accessed)
            .await
            .unwrap();
        assert_eq!(store.get(&a).await.unwrap().unwrap().access_count, 1);
        assert_eq!(store.get(&b).await.unwrap().unwrap().access_count, 1);

        // Procedural outcome also works in batch (feeds Wilson).
        let p = store
            .remember(Memory::procedural(
                s.clone(),
                "compile",
                vec!["cargo test".into()],
            ))
            .await
            .unwrap();
        store
            .reinforce_many(std::slice::from_ref(&p), Outcome::Success)
            .await
            .unwrap();
        let pm = store.get(&p).await.unwrap().unwrap();
        assert!(pm.summary().contains("1\u{2713}/0\u{2717}"));
    }

    #[tokio::test]
    async fn reinforce_many_atomicity() {
        // Fix 7: reinforce_many runs inside a single transaction.
        // If it completes, all records are updated atomically.
        let store = SqliteStore::in_memory().unwrap();
        let s = scope();
        let a = store
            .remember(Memory::semantic(s.clone(), "alpha", SemanticCat::Fact))
            .await
            .unwrap();
        let b = store
            .remember(Memory::semantic(s.clone(), "beta", SemanticCat::Fact))
            .await
            .unwrap();
        store
            .reinforce_many(&[a.clone(), b.clone()], Outcome::Accessed)
            .await
            .unwrap();
        let ma = store.get(&a).await.unwrap().unwrap();
        let mb = store.get(&b).await.unwrap().unwrap();
        assert_eq!(ma.access_count, 1, "a reinforced");
        assert_eq!(mb.access_count, 1, "b reinforced");
        // Both timestamps should be identical (captured before loop).
        assert_eq!(
            ma.last_access, mb.last_access,
            "both updated with the same `now` timestamp in one transaction"
        );
    }

    #[tokio::test]
    async fn graph_leg_pulls_entity_bridge_neighbor_and_persists() {
        let db = TmpDb::new();
        let scope = scope();
        {
            let store = SqliteStore::open(&db.0).unwrap();
            store
                .remember(Memory::semantic(
                    scope.clone(),
                    "Aylin adopted a tabby cat and named it Paspas",
                    SemanticCat::Fact,
                ))
                .await
                .unwrap();
            store
                .remember(Memory::semantic(
                    scope.clone(),
                    "Paspas was vaccinated at the veterinary clinic",
                    SemanticCat::Fact,
                ))
                .await
                .unwrap();
        }
        // Reopen: the entity index must survive restarts (it is a table, not
        // a cache) and feed the graph leg identically.
        let store = SqliteStore::open(&db.0).unwrap();
        let res = store
            .recall(&scope, &Query::new("aylin cat").graph().limit(5))
            .await
            .unwrap();
        let pulled = res
            .iter()
            .find(|s| s.item.searchable_text().contains("vaccinated"))
            .expect("graph leg should pull the vaccination record after reopen");
        assert!(pulled.signals.iter().any(|s| s.name == "graph"));
    }

    #[tokio::test]
    async fn schema_v3_backfills_entities_for_v2_files() {
        let db = TmpDb::new();
        let scope = scope();
        // Build a store, then simulate a pre-v3 file: drop the entities
        // table and stamp schema=2. Opening must recreate + backfill.
        {
            let store = SqliteStore::open(&db.0).unwrap();
            store
                .remember(Memory::semantic(
                    scope.clone(),
                    "Paspas was vaccinated at the veterinary clinic",
                    SemanticCat::Fact,
                ))
                .await
                .unwrap();
            let conn = store.conn.lock().unwrap();
            conn.execute_batch(
                "DROP TABLE entities;
                 UPDATE meta SET value = '2' WHERE key = 'schema';",
            )
            .unwrap();
        }
        let store = SqliteStore::open(&db.0).unwrap();
        let n: i64 = store
            .conn
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
            .unwrap();
        assert!(n > 0, "v2→v3 open must backfill the entities table");
        assert_eq!(
            SqliteStore::meta_get(&store.conn.lock().unwrap(), "schema")
                .unwrap()
                .as_deref(),
            Some("3")
        );
    }

    #[tokio::test]
    async fn reinforce_and_forget_persist() {
        let store = SqliteStore::in_memory().unwrap();
        let s = scope();
        let id = store
            .remember(Memory::procedural(
                s.clone(),
                "compile and test",
                vec!["cargo test".into()],
            ))
            .await
            .unwrap();
        store.reinforce(&id, Outcome::Success).await.unwrap();

        let res = store.recall(&s, &Query::new("compile")).await.unwrap();
        assert!(res[0].item.summary().contains("1✓/0✗"));

        store.forget(&id).await.unwrap();
        assert_eq!(
            store
                .recall(&s, &Query::new("compile"))
                .await
                .unwrap()
                .len(),
            0
        );
        assert_eq!(store.total_rows().unwrap(), 1); // soft-delete: row stays
        assert_eq!(store.count(&s).await.unwrap(), 0); // live count: deleted dropped
    }

    #[tokio::test]
    async fn v1_migration_skips_corrupt_rows_and_keeps_good() {
        // A v1 DB with corrupt JSON in one row should NOT kill the migration.
        // The corrupt row is skipped (warned); valid rows are preserved.
        let tmp = TmpDb::new();
        let _e = HashingEmbedder::new();
        let s = scope();
        {
            let conn = Connection::open(&tmp.0).unwrap();
            conn.execute_batch(
                "CREATE TABLE memories (
                    id TEXT PRIMARY KEY, scope_agent TEXT, is_world INTEGER NOT NULL,
                    tier TEXT NOT NULL, deleted INTEGER NOT NULL, data TEXT NOT NULL);
                 CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
            )
            .unwrap();
            // Good row: valid JSON memory.
            let good = Memory::semantic(s.clone(), "Rust is fast", SemanticCat::Fact);
            let agent = match &good.scope {
                Scope::Agent(a) => Some(a.to_string()),
                Scope::World => None,
            };
            conn.execute(
                "INSERT INTO memories (id, scope_agent, is_world, tier, deleted, data)
                 VALUES (?1, ?2, 0, 'Semantic', 0, ?3)",
                params![
                    good.id.to_string(),
                    agent,
                    serde_json::to_string(&good).unwrap()
                ],
            )
            .unwrap();
            // Corrupt row: invalid JSON that cannot deserialize to Memory.
            conn.execute(
                "INSERT INTO memories (id, scope_agent, is_world, tier, deleted, data)
                 VALUES ('corrupt-id', 'bad-agent', 0, 'Semantic', 0, '{not valid json!!!')",
                [],
            )
            .unwrap();
        }

        // Migration completes (does not panic / return Err).
        let store = SqliteStore::open(&tmp.0).unwrap();
        // Good row is preserved and searchable via FTS.
        let hits = store.recall(&s, &Query::new("rust")).await.unwrap();
        assert_eq!(
            hits.len(),
            1,
            "valid row preserved after migration with corrupt neighbor"
        );
        // Corrupt row was NOT updated (still has old data, no search_text);
        // it stays in the DB but decode_row will fail on recall — that is
        // acceptable: the row is effectively inert (no FTS match).
        assert_eq!(
            store.total_rows().unwrap(),
            2,
            "both rows still exist in DB"
        );
    }

    #[tokio::test]
    async fn consolidate_survives_concurrent_writes() {
        // Soak regression: consolidation reads on a separate connection then
        // writes; if the main connection writes in between, a deferred
        // transaction would immediately fail with SQLITE_BUSY_SNAPSHOT
        // (busy_timeout has no effect). BEGIN IMMEDIATE acquires the write
        // lock upfront — no error under concurrent writes.
        let tmp = super::tests::TmpDb::new();
        let store = Arc::new(
            SqliteStore::open(&tmp.0)
                .unwrap()
                .with_embedder(Arc::new(HashingEmbedder::new())),
        );
        let s = scope();
        // Writer: writes continuously while consolidation runs.
        let writer_store = store.clone();
        let ws = s.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_w = stop.clone();
        let writer = tokio::spawn(async move {
            let mut i = 0;
            while !stop_w.load(std::sync::atomic::Ordering::Relaxed) {
                // The writer is load, not the assertion target: while
                // consolidation holds BEGIN IMMEDIATE, transient
                // "database is locked" errors are expected — retry
                // briefly instead of panicking (this was the flake).
                let mut attempts = 0;
                loop {
                    match writer_store
                        .remember(Memory::episodic(
                            ws.clone(),
                            format!("entry {i}"),
                            "concurrent",
                        ))
                        .await
                    {
                        Ok(_) => break,
                        Err(e) if e.to_string().contains("locked") && attempts < 200 => {
                            attempts += 1;
                            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                        }
                        Err(e) => panic!("writer failed with non-lock error: {e}"),
                    }
                }
                i += 1;
            }
        });
        // Let the writer fill up a bit.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        for round in 0..5 {
            store
                .consolidate()
                .await
                .unwrap_or_else(|e| panic!("consolidation round {round} exploded: {e}"));
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn load_by_ids_chunking_with_over_500_ids() {
        // Edge test: >500 ids triggers chunking — all results returned in
        // input order, missing ids silently skipped.
        let store = SqliteStore::in_memory().unwrap();
        let s = scope();
        // Insert 503 records.
        let mut ids: Vec<String> = Vec::with_capacity(503);
        for i in 0..503 {
            let id = store
                .remember(Memory::semantic(
                    s.clone(),
                    format!("item {i}"),
                    SemanticCat::Fact,
                ))
                .await
                .unwrap();
            ids.push(id.to_string());
        }
        // Shuffle: put a missing id at position 0, reverse the rest.
        let mut request_ids: Vec<String> = vec![MemoryId::new().to_string()];
        request_ids.extend(ids.iter().rev().cloned());

        let conn = store.conn.lock().unwrap();
        let loaded = SqliteStore::load_by_ids(&conn, &request_ids).unwrap();
        assert_eq!(
            loaded.len(),
            503,
            "all 503 live records returned, missing skipped"
        );
        // Verify ordering: should follow request_ids, skipping the missing one.
        assert_eq!(
            loaded[0].id.to_string(),
            ids[502],
            "first = last inserted (reversed)"
        );
        assert_eq!(
            loaded[502].id.to_string(),
            ids[0],
            "last = first inserted (reversed)"
        );
    }
}
