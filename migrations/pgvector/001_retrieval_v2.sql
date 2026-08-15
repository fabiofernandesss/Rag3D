-- Retrieval Engine V2 pgvector schema (idempotent operator migration).
-- The vector extension remains infrastructure-owned and is never created here.
--
--   psql "$RAG3D_PG" -v dense_dim=1024 -v structural_dim=128 \
--     -f migrations/pgvector/001_retrieval_v2.sql

\set ON_ERROR_STOP on

\if :{?dense_dim}
\else
  \echo 'dense_dim is required (psql -v dense_dim=1024)'
  DO $$ BEGIN RAISE EXCEPTION 'dense_dim is required'; END $$;
\endif

\if :{?structural_dim}
\else
  \echo 'structural_dim is required (psql -v structural_dim=128)'
  DO $$ BEGIN RAISE EXCEPTION 'structural_dim is required'; END $$;
\endif

BEGIN;
SET LOCAL search_path = pg_catalog, public;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
SELECT pg_advisory_xact_lock(x'5241473344563201'::bit(64)::bigint);

-- Validate psql substitutions before any statement interpolates them as SQL.
SELECT set_config('rag3d.migration_dense_dim', :'dense_dim', true);
SELECT set_config('rag3d.migration_structural_dim', :'structural_dim', true);
DO $$
DECLARE
  dense_text TEXT := current_setting('rag3d.migration_dense_dim');
  structural_text TEXT := current_setting('rag3d.migration_structural_dim');
BEGIN
  IF dense_text !~ '^[0-9]+$' OR dense_text::numeric NOT BETWEEN 1 AND 2000 THEN
    RAISE EXCEPTION 'dense_dim must be an integer between 1 and 2000';
  END IF;
  IF structural_text !~ '^[0-9]+$' OR
     structural_text::numeric NOT BETWEEN 1 AND 2000 THEN
    RAISE EXCEPTION 'structural_dim must be an integer between 1 and 2000';
  END IF;
  PERFORM set_config(
    'rag3d.migration_dense_dim', dense_text::integer::text, true
  );
  PERFORM set_config(
    'rag3d.migration_structural_dim', structural_text::integer::text, true
  );
END
$$;

CREATE TABLE IF NOT EXISTS public.rag3d_v2_meta(
  key TEXT CONSTRAINT rag3d_v2_meta_pk PRIMARY KEY,
  value TEXT NOT NULL
);

DO $$
DECLARE
  stored_dense TEXT;
  stored_structural TEXT;
BEGIN
  SELECT value INTO stored_dense
    FROM public.rag3d_v2_meta WHERE key = 'dense_dim';
  SELECT value INTO stored_structural
    FROM public.rag3d_v2_meta WHERE key = 'structural_dim';
  IF stored_dense IS NOT NULL AND
     stored_dense <> current_setting('rag3d.migration_dense_dim') THEN
    RAISE EXCEPTION 'incompatible pgvector schema state for dense_dim';
  END IF;
  IF stored_structural IS NOT NULL AND
     stored_structural <> current_setting('rag3d.migration_structural_dim') THEN
    RAISE EXCEPTION 'incompatible pgvector schema state for structural_dim';
  END IF;
END
$$;

CREATE TABLE IF NOT EXISTS public.rag3d_v2_documents(
  id BIGSERIAL CONSTRAINT rag3d_v2_documents_pk PRIMARY KEY,
  source TEXT NOT NULL,
  title TEXT NOT NULL,
  created DOUBLE PRECISION NOT NULL,
  n_tokens INTEGER NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT rag3d_v2_documents_tokens_ck CHECK (n_tokens >= 0)
);

CREATE TABLE IF NOT EXISTS public.rag3d_v2_chunks(
  id BIGSERIAL CONSTRAINT rag3d_v2_chunks_pk PRIMARY KEY,
  document_id BIGINT,
  parent_id BIGINT,
  kind TEXT NOT NULL DEFAULT 'chunk',
  pos INTEGER NOT NULL DEFAULT 0,
  text TEXT NOT NULL,
  context TEXT NOT NULL DEFAULT '',
  n_tokens INTEGER NOT NULL,
  created DOUBLE PRECISION NOT NULL,
  importance REAL NOT NULL DEFAULT 0.5,
  turn_no INTEGER,
  accessed_turn INTEGER,
  embedding vector(:dense_dim),
  structural_n_tok INTEGER,
  structural_dim INTEGER,
  structural_data BYTEA,
  CONSTRAINT rag3d_v2_chunks_document_fk FOREIGN KEY(document_id)
    REFERENCES public.rag3d_v2_documents(id) ON DELETE CASCADE,
  CONSTRAINT rag3d_v2_chunks_parent_fk FOREIGN KEY(parent_id)
    REFERENCES public.rag3d_v2_chunks(id) ON DELETE SET NULL,
  CONSTRAINT rag3d_v2_chunks_kind_ck CHECK (
    kind IN ('chunk','parent','summary','turn','rolling_summary')
  ),
  CONSTRAINT rag3d_v2_chunks_position_ck CHECK (
    pos >= 0 OR (kind = 'summary' AND pos = -1)
  ),
  CONSTRAINT rag3d_v2_chunks_tokens_ck CHECK (n_tokens >= 0),
  CONSTRAINT rag3d_v2_chunks_importance_ck CHECK (
    importance >= 0 AND importance <= 1
  ),
  CONSTRAINT rag3d_v2_chunks_embedding_ck CHECK (
    (kind = 'parent' AND embedding IS NULL) OR
    (kind <> 'parent' AND embedding IS NOT NULL)
  ),
  CONSTRAINT rag3d_v2_chunks_structural_ck CHECK (
    (kind = 'parent' AND structural_n_tok IS NULL AND
     structural_dim IS NULL AND structural_data IS NULL) OR
    (kind <> 'parent' AND structural_n_tok > 0 AND
     structural_dim = :structural_dim AND structural_data IS NOT NULL AND
     octet_length(structural_data) = structural_n_tok * structural_dim * 2)
  )
);

-- Upgrade a pre-release 001 schema created before the position invariant.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_constraint
    WHERE conname = 'rag3d_v2_chunks_position_ck'
      AND conrelid = 'public.rag3d_v2_chunks'::regclass
  ) THEN
    ALTER TABLE public.rag3d_v2_chunks
      ADD CONSTRAINT rag3d_v2_chunks_position_ck CHECK (
        pos >= 0 OR (kind = 'summary' AND pos = -1)
      );
  END IF;
END
$$;

CREATE TABLE IF NOT EXISTS public.rag3d_v2_sparse_postings(
  term BIGINT NOT NULL,
  chunk_id BIGINT NOT NULL,
  weight REAL NOT NULL,
  CONSTRAINT rag3d_v2_sparse_postings_pk PRIMARY KEY(term, chunk_id),
  CONSTRAINT rag3d_v2_sparse_postings_chunk_fk FOREIGN KEY(chunk_id)
    REFERENCES public.rag3d_v2_chunks(id) ON DELETE CASCADE,
  CONSTRAINT rag3d_v2_sparse_postings_weight_ck CHECK (
    weight <> 'NaN'::real AND
    weight > '-Infinity'::real AND weight < 'Infinity'::real
  )
);

CREATE INDEX IF NOT EXISTS rag3d_v2_documents_source_idx
  ON public.rag3d_v2_documents(source);
CREATE INDEX IF NOT EXISTS rag3d_v2_chunks_document_idx
  ON public.rag3d_v2_chunks(document_id);
CREATE INDEX IF NOT EXISTS rag3d_v2_chunks_parent_idx
  ON public.rag3d_v2_chunks(parent_id);
CREATE INDEX IF NOT EXISTS rag3d_v2_chunks_kind_idx
  ON public.rag3d_v2_chunks(kind);
CREATE INDEX IF NOT EXISTS rag3d_v2_chunks_turn_idx
  ON public.rag3d_v2_chunks(turn_no);
CREATE INDEX IF NOT EXISTS rag3d_v2_sparse_postings_chunk_idx
  ON public.rag3d_v2_sparse_postings(chunk_id);

INSERT INTO public.rag3d_v2_meta(key, value)
VALUES
  ('schema_version', '1'),
  ('dense_dim', current_setting('rag3d.migration_dense_dim')),
  ('structural_dim', current_setting('rag3d.migration_structural_dim')),
  ('normalization', 'l2'),
  ('quantization', 'none')
ON CONFLICT (key) DO NOTHING;

-- IF NOT EXISTS is not verification: reject drift in dimensions, state,
-- constraints, and every required non-partial/non-unique B-tree index.
DO $$
DECLARE
  base_index_count INTEGER;
  required_constraint_count INTEGER;
  required_column_count INTEGER;
  actual_column_count INTEGER;
BEGIN
  WITH expected(
    table_name, column_name, data_type, not_null,
    default_contract, identity_kind
  ) AS (VALUES
    ('rag3d_v2_meta','key','text',TRUE,'',''),
    ('rag3d_v2_meta','value','text',TRUE,'',''),
    ('rag3d_v2_documents','id','bigint',TRUE,'serial',''),
    ('rag3d_v2_documents','source','text',TRUE,'',''),
    ('rag3d_v2_documents','title','text',TRUE,'',''),
    ('rag3d_v2_documents','created','double precision',TRUE,'',''),
    ('rag3d_v2_documents','n_tokens','integer',TRUE,'',''),
    ('rag3d_v2_documents','metadata','jsonb',TRUE,
      $default$'{}'::jsonb$default$,''),
    ('rag3d_v2_chunks','id','bigint',TRUE,'serial',''),
    ('rag3d_v2_chunks','document_id','bigint',FALSE,'',''),
    ('rag3d_v2_chunks','parent_id','bigint',FALSE,'',''),
    ('rag3d_v2_chunks','kind','text',TRUE,
      $default$'chunk'::text$default$,''),
    ('rag3d_v2_chunks','pos','integer',TRUE,'0',''),
    ('rag3d_v2_chunks','text','text',TRUE,'',''),
    ('rag3d_v2_chunks','context','text',TRUE,
      $default$''::text$default$,''),
    ('rag3d_v2_chunks','n_tokens','integer',TRUE,'',''),
    ('rag3d_v2_chunks','created','double precision',TRUE,'',''),
    ('rag3d_v2_chunks','importance','real',TRUE,'0.5',''),
    ('rag3d_v2_chunks','turn_no','integer',FALSE,'',''),
    ('rag3d_v2_chunks','accessed_turn','integer',FALSE,'',''),
    ('rag3d_v2_chunks','embedding',
      'vector(' || current_setting('rag3d.migration_dense_dim') || ')',
      FALSE,'',''),
    ('rag3d_v2_chunks','structural_n_tok','integer',FALSE,'',''),
    ('rag3d_v2_chunks','structural_dim','integer',FALSE,'',''),
    ('rag3d_v2_chunks','structural_data','bytea',FALSE,'',''),
    ('rag3d_v2_sparse_postings','term','bigint',TRUE,'',''),
    ('rag3d_v2_sparse_postings','chunk_id','bigint',TRUE,'',''),
    ('rag3d_v2_sparse_postings','weight','real',TRUE,'','')
  ), actual_raw AS (
    SELECT cls.relname AS table_name,
           attr.attname AS column_name,
           pg_catalog.format_type(attr.atttypid, attr.atttypmod) AS data_type,
           attr.attnotnull AS not_null,
           regexp_replace(
             lower(COALESCE(pg_catalog.pg_get_expr(
               def.adbin, def.adrelid, false
             ), '')),
             '\s+', '', 'g'
           ) AS default_expression,
           attr.attidentity::text AS identity_kind,
           COALESCE(
             pg_catalog.pg_get_serial_sequence(
               pg_catalog.format('%I.%I', ns.nspname, cls.relname),
               attr.attname
             ),
             ''
           ) AS serial_sequence
    FROM pg_catalog.pg_class AS cls
    JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
    JOIN pg_catalog.pg_attribute AS attr ON attr.attrelid = cls.oid
    LEFT JOIN pg_catalog.pg_attrdef AS def
      ON def.adrelid = attr.attrelid AND def.adnum = attr.attnum
    WHERE ns.nspname = 'public'
      AND cls.relkind = 'r'
      AND cls.relname = ANY(ARRAY[
        'rag3d_v2_meta','rag3d_v2_documents','rag3d_v2_chunks',
        'rag3d_v2_sparse_postings'
      ])
      AND attr.attnum > 0
      AND NOT attr.attisdropped
  ), actual AS (
    SELECT table_name, column_name, data_type, not_null,
           CASE
             WHEN (table_name, column_name) IN (
                    ('rag3d_v2_documents','id'),
                    ('rag3d_v2_chunks','id')
                  )
              AND identity_kind = ''
              AND serial_sequence =
                    'public.' || table_name || '_id_seq'
              AND default_expression IN (
                    'nextval(''' || table_name || '_id_seq''::regclass)',
                    'nextval(''public.' || table_name || '_id_seq''::regclass)'
                  )
             THEN 'serial'
             ELSE default_expression
           END AS default_contract,
           identity_kind
    FROM actual_raw
  )
  SELECT
    (SELECT COUNT(*)
     FROM expected
     JOIN actual USING (
       table_name, column_name, data_type, not_null,
       default_contract, identity_kind
     )),
    (SELECT COUNT(*) FROM actual)
  INTO required_column_count, actual_column_count;
  IF required_column_count <> 27 OR actual_column_count <> 27 THEN
    RAISE EXCEPTION 'incompatible pgvector column catalog';
  END IF;

  IF pg_catalog.format_type(
       (SELECT atttypid FROM pg_catalog.pg_attribute
        WHERE attrelid = 'public.rag3d_v2_chunks'::regclass
          AND attname = 'embedding' AND NOT attisdropped),
       (SELECT atttypmod FROM pg_catalog.pg_attribute
        WHERE attrelid = 'public.rag3d_v2_chunks'::regclass
          AND attname = 'embedding' AND NOT attisdropped)
     ) <> 'vector(' || current_setting('rag3d.migration_dense_dim') || ')' THEN
    RAISE EXCEPTION 'incompatible pgvector embedding dimension';
  END IF;

  WITH expected(
    constraint_name, table_name, constraint_type, columns,
    target_table, target_columns, delete_action, expression
  ) AS (VALUES
    ('rag3d_v2_meta_pk','rag3d_v2_meta','p',ARRAY['key']::text[],
      '',ARRAY[]::text[],'',''),
    ('rag3d_v2_documents_pk','rag3d_v2_documents','p',ARRAY['id']::text[],
      '',ARRAY[]::text[],'',''),
    ('rag3d_v2_documents_tokens_ck','rag3d_v2_documents','c',
      ARRAY['n_tokens']::text[],'',ARRAY[]::text[],'','(n_tokens>=0)'),
    ('rag3d_v2_chunks_pk','rag3d_v2_chunks','p',ARRAY['id']::text[],
      '',ARRAY[]::text[],'',''),
    ('rag3d_v2_chunks_document_fk','rag3d_v2_chunks','f',
      ARRAY['document_id']::text[],'rag3d_v2_documents',ARRAY['id']::text[],
      'c',''),
    ('rag3d_v2_chunks_parent_fk','rag3d_v2_chunks','f',
      ARRAY['parent_id']::text[],'rag3d_v2_chunks',ARRAY['id']::text[],
      'n',''),
    ('rag3d_v2_chunks_kind_ck','rag3d_v2_chunks','c',ARRAY['kind']::text[],
      '',ARRAY[]::text[],'',
      '(kind=any(array[''chunk''::text,''parent''::text,''summary''::text,' ||
      '''turn''::text,''rolling_summary''::text]))'),
    ('rag3d_v2_chunks_position_ck','rag3d_v2_chunks','c',
      ARRAY['pos','kind']::text[],'',ARRAY[]::text[],'',
      '((pos>=0)or((kind=''summary''::text)and(pos=''-1''::integer)))'),
    ('rag3d_v2_chunks_' || 'to' || 'kens_ck','rag3d_v2_chunks','c',
      ARRAY['n_tokens']::text[],'',ARRAY[]::text[],'','(n_tokens>=0)'),
    ('rag3d_v2_chunks_importance_ck','rag3d_v2_chunks','c',
      ARRAY['importance']::text[],'',ARRAY[]::text[],'',
      '((importance>=(0)::doubleprecision)and' ||
      '(importance<=(1)::doubleprecision))'),
    ('rag3d_v2_chunks_embedding_ck','rag3d_v2_chunks','c',
      ARRAY['kind','embedding']::text[],'',ARRAY[]::text[],'',
      '(((kind=''parent''::text)and(embeddingisnull))or' ||
      '((kind<>''parent''::text)and(embeddingisnotnull)))'),
    ('rag3d_v2_chunks_structural_ck','rag3d_v2_chunks','c',
      ARRAY['kind','structural_n_tok','structural_dim','structural_data']::text[],
      '',ARRAY[]::text[],'',
      '(((kind=''parent''::text)and(structural_n_tokisnull)and' ||
      '(structural_dimisnull)and(structural_dataisnull))or' ||
      '((kind<>''parent''::text)and(structural_n_tok>0)and' ||
      '(structural_dim=' || current_setting('rag3d.migration_structural_dim') ||
      ')and(structural_dataisnotnull)and(octet_length(structural_data)=' ||
      '((structural_n_tok*structural_dim)*2))))'),
    ('rag3d_v2_sparse_postings_pk','rag3d_v2_sparse_postings','p',
      ARRAY['term','chunk_id']::text[],'',ARRAY[]::text[],'',''),
    ('rag3d_v2_sparse_postings_chunk_fk','rag3d_v2_sparse_postings','f',
      ARRAY['chunk_id']::text[],'rag3d_v2_chunks',ARRAY['id']::text[],'c',''),
    ('rag3d_v2_sparse_postings_weight_ck','rag3d_v2_sparse_postings','c',
      ARRAY['weight']::text[],'',ARRAY[]::text[],'',
      '((weight<>''nan''::real)and(weight>''-infinity''::real)and' ||
      '(weight<''infinity''::real))')
  ), actual AS (
    SELECT con.conname AS constraint_name,
           cls.relname AS table_name,
           con.contype::text AS constraint_type,
           con.convalidated,
           COALESCE((
             SELECT array_agg(attr.attname::text ORDER BY key.ord)
             FROM unnest(con.conkey) WITH ORDINALITY AS key(attnum,ord)
             JOIN pg_catalog.pg_attribute AS attr
               ON attr.attrelid = con.conrelid AND attr.attnum = key.attnum
           ), ARRAY[]::text[]) AS columns,
           CASE
             WHEN target.oid IS NULL THEN ''
             WHEN target_ns.nspname = 'public' THEN target.relname
             ELSE target_ns.nspname || '.' || target.relname
           END AS target_table,
           COALESCE((
             SELECT array_agg(attr.attname::text ORDER BY key.ord)
             FROM unnest(con.confkey) WITH ORDINALITY AS key(attnum,ord)
             JOIN pg_catalog.pg_attribute AS attr
               ON attr.attrelid = con.confrelid AND attr.attnum = key.attnum
           ), ARRAY[]::text[]) AS target_columns,
           CASE WHEN con.contype = 'f' THEN con.confdeltype::text ELSE '' END
             AS delete_action,
           regexp_replace(
             lower(COALESCE(pg_catalog.pg_get_expr(
               con.conbin, con.conrelid, false
             ), '')),
             '\s+', '', 'g'
           ) AS expression
    FROM pg_catalog.pg_constraint AS con
    JOIN pg_catalog.pg_class AS cls ON cls.oid = con.conrelid
    JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
    LEFT JOIN pg_catalog.pg_class AS target ON target.oid = con.confrelid
    LEFT JOIN pg_catalog.pg_namespace AS target_ns
      ON target_ns.oid = target.relnamespace
    WHERE ns.nspname = 'public'
      AND cls.relname = ANY(ARRAY[
        'rag3d_v2_meta','rag3d_v2_documents','rag3d_v2_chunks',
        'rag3d_v2_sparse_postings'
      ])
  )
  SELECT COUNT(*) INTO required_constraint_count
  FROM expected
  JOIN actual USING (
    constraint_name, table_name, constraint_type, columns,
    target_table, target_columns, delete_action, expression
  )
  WHERE actual.convalidated;
  IF required_constraint_count <> 15 THEN
    RAISE EXCEPTION 'incompatible pgvector constraint catalog';
  END IF;

  WITH expected(index_name, table_name, column_name) AS (VALUES
    ('rag3d_v2_documents_source_idx','rag3d_v2_documents','source'),
    ('rag3d_v2_chunks_document_idx','rag3d_v2_chunks','document_id'),
    ('rag3d_v2_chunks_parent_idx','rag3d_v2_chunks','parent_id'),
    ('rag3d_v2_chunks_kind_idx','rag3d_v2_chunks','kind'),
    ('rag3d_v2_chunks_turn_idx','rag3d_v2_chunks','turn_no'),
    ('rag3d_v2_sparse_postings_chunk_idx',
      'rag3d_v2_sparse_postings','chunk_id')
  )
  SELECT COUNT(*) INTO base_index_count
  FROM expected
  JOIN pg_catalog.pg_class idx ON idx.relname = expected.index_name
  JOIN pg_catalog.pg_namespace ns
    ON ns.oid = idx.relnamespace AND ns.nspname = 'public'
  JOIN pg_catalog.pg_index ind ON ind.indexrelid = idx.oid
  JOIN pg_catalog.pg_class tbl
    ON tbl.oid = ind.indrelid AND tbl.relname = expected.table_name
  JOIN pg_catalog.pg_am am ON am.oid = idx.relam AND am.amname = 'btree'
  JOIN pg_catalog.pg_attribute attr
    ON attr.attrelid = tbl.oid AND attr.attnum = ind.indkey[0]
   AND attr.attname = expected.column_name
  WHERE ind.indnatts = 1 AND ind.indisvalid AND ind.indisready
    AND NOT ind.indisunique AND ind.indpred IS NULL;
  IF base_index_count <> 6 THEN
    RAISE EXCEPTION 'incompatible pgvector base index catalog';
  END IF;

  IF EXISTS (
    SELECT 1
    FROM (VALUES
      ('schema_version','1'),
      ('dense_dim',current_setting('rag3d.migration_dense_dim')),
      ('structural_dim',current_setting('rag3d.migration_structural_dim')),
      ('normalization','l2'),('quantization','none')
    ) AS expected(key,value)
    LEFT JOIN public.rag3d_v2_meta actual USING (key)
    WHERE actual.value IS DISTINCT FROM expected.value
  ) THEN
    RAISE EXCEPTION 'incompatible pgvector schema state';
  END IF;
END
$$;

COMMIT;
