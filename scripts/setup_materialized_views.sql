-- Materialized View: mv_daily_sff
-- 3-table JOIN + GROUP BY + DISTINCT ON을 사전 계산하여 쿼리 성능 대폭 개선
-- (19초 → ~1-2초)
--
-- 사용법:
--   psql -U postgres -d korea_stock_data -f scripts/setup_materialized_views.sql
--
-- 리프레시 (장 마감 후 데이터 업데이트 시):
--   REFRESH MATERIALIZED VIEW mv_daily_sff;

-- 기존 MV 삭제 (재생성 시)
DROP MATERIALIZED VIEW IF EXISTS mv_daily_sff;

CREATE MATERIALIZED VIEW mv_daily_sff AS
SELECT
    it.time AS trade_date,
    it.stock_code,
    SUM(CASE WHEN it.investor_type = 'FOREIGN' THEN it.net_buy_value ELSE 0 END) AS foreign_net_amount,
    SUM(CASE WHEN it.investor_type = 'INSTITUTION' THEN it.net_buy_value ELSE 0 END) AS institution_net_amount,
    MAX(o.close_price) AS close_price,
    ff.floating_shares AS free_float_shares,
    -- Sff 사전 계산 (individual, weight-independent)
    CASE WHEN MAX(o.close_price) * ff.floating_shares > 0
         THEN (SUM(CASE WHEN it.investor_type = 'FOREIGN' THEN it.net_buy_value ELSE 0 END)::float
               / (MAX(o.close_price) * ff.floating_shares)) * 100
         ELSE 0 END AS foreign_sff,
    CASE WHEN MAX(o.close_price) * ff.floating_shares > 0
         THEN (SUM(CASE WHEN it.investor_type = 'INSTITUTION' THEN it.net_buy_value ELSE 0 END)::float
               / (MAX(o.close_price) * ff.floating_shares)) * 100
         ELSE 0 END AS institution_sff
FROM investor_trading it
JOIN ohlcv_daily o ON it.time = o.time AND it.stock_code = o.stock_code
JOIN (
    SELECT DISTINCT ON (stock_code) stock_code, floating_shares
    FROM floating_shares
    ORDER BY stock_code, base_date DESC
) ff ON it.stock_code = ff.stock_code
WHERE it.investor_type IN ('FOREIGN', 'INSTITUTION')
  AND o.close_price IS NOT NULL
GROUP BY it.time, it.stock_code, ff.floating_shares;

-- 인덱스
CREATE INDEX idx_mv_sff_stock_date ON mv_daily_sff (stock_code, trade_date DESC);
CREATE INDEX idx_mv_sff_date ON mv_daily_sff (trade_date DESC);

-- 읽기 권한
GRANT SELECT ON mv_daily_sff TO korea_stock_reader;

-- 확인
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT stock_code) AS total_stocks,
       MIN(trade_date) AS min_date,
       MAX(trade_date) AS max_date
FROM mv_daily_sff;


-- =========================================================================
-- SQL 함수: fn_zscore_latest — 최신(또는 지정) 날짜의 7개 기간 Z-Score 계산
-- LATERAL + FILTER 패턴으로 Python 28초 → SQL 0.7초 (40배 향상)
--
-- 사용법:
--   SELECT * FROM fn_zscore_latest(0.3);              -- 기관가중치=0.3, 최신 날짜
--   SELECT * FROM fn_zscore_latest(0.5, '2026-02-27'); -- 기관가중치=0.5, 특정 날짜
-- =========================================================================

DROP FUNCTION IF EXISTS fn_zscore_latest(float, date);

CREATE OR REPLACE FUNCTION fn_zscore_latest(
    w_inst float DEFAULT 0.3,
    target_date date DEFAULT NULL
) RETURNS TABLE (
    stock_code varchar,
    today_sff double precision,
    foreign_sff double precision,
    institution_sff double precision,
    foreign_net_amount double precision,
    institution_net_amount double precision,
    sff_5d_avg double precision,
    std_5d double precision, std_10d double precision, std_20d double precision,
    std_50d double precision, std_100d double precision, std_200d double precision,
    std_500d double precision,
    zscore_5d double precision, zscore_10d double precision, zscore_20d double precision,
    zscore_50d double precision, zscore_100d double precision, zscore_200d double precision,
    zscore_500d double precision
) AS $$
WITH ref AS (
    SELECT COALESCE(target_date, MAX(m.trade_date)) AS ref_date
    FROM mv_daily_sff m
),
stocks_on_date AS (
    SELECT m.stock_code,
           m.foreign_sff AS f_sff,
           m.institution_sff AS i_sff,
           m.foreign_net_amount,
           m.institution_net_amount,
           CASE WHEN (m.foreign_sff * m.institution_sff) > 0
                THEN m.foreign_sff + m.institution_sff * w_inst
                ELSE m.foreign_sff END AS today_sff
    FROM mv_daily_sff m, ref
    WHERE m.trade_date = ref.ref_date
)
SELECT s.stock_code,
       s.today_sff,
       s.f_sff,
       s.i_sff,
       s.foreign_net_amount,
       s.institution_net_amount,
       z.sff_5d_avg,
       z.std_5d, z.std_10d, z.std_20d, z.std_50d,
       z.std_100d, z.std_200d, z.std_500d,
       -- 조건부 Z-Score: 부호 전환 시 과잉 반응 방지
       CASE WHEN s.today_sff * z.mean_5d > 0
            THEN (s.today_sff - z.mean_5d) / NULLIF(z.std_5d, 0)
            ELSE s.today_sff / NULLIF(z.std_5d, 0) END,
       CASE WHEN s.today_sff * z.mean_10d > 0
            THEN (s.today_sff - z.mean_10d) / NULLIF(z.std_10d, 0)
            ELSE s.today_sff / NULLIF(z.std_10d, 0) END,
       CASE WHEN s.today_sff * z.mean_20d > 0
            THEN (s.today_sff - z.mean_20d) / NULLIF(z.std_20d, 0)
            ELSE s.today_sff / NULLIF(z.std_20d, 0) END,
       CASE WHEN s.today_sff * z.mean_50d > 0
            THEN (s.today_sff - z.mean_50d) / NULLIF(z.std_50d, 0)
            ELSE s.today_sff / NULLIF(z.std_50d, 0) END,
       CASE WHEN s.today_sff * z.mean_100d > 0
            THEN (s.today_sff - z.mean_100d) / NULLIF(z.std_100d, 0)
            ELSE s.today_sff / NULLIF(z.std_100d, 0) END,
       CASE WHEN s.today_sff * z.mean_200d > 0
            THEN (s.today_sff - z.mean_200d) / NULLIF(z.std_200d, 0)
            ELSE s.today_sff / NULLIF(z.std_200d, 0) END,
       CASE WHEN s.today_sff * z.mean_500d > 0
            THEN (s.today_sff - z.mean_500d) / NULLIF(z.std_500d, 0)
            ELSE s.today_sff / NULLIF(z.std_500d, 0) END
FROM stocks_on_date s
CROSS JOIN LATERAL (
    SELECT
        AVG(cs) FILTER (WHERE rn <= 5)   AS mean_5d,
        AVG(cs) FILTER (WHERE rn <= 10)  AS mean_10d,
        AVG(cs) FILTER (WHERE rn <= 20)  AS mean_20d,
        AVG(cs) FILTER (WHERE rn <= 50)  AS mean_50d,
        AVG(cs) FILTER (WHERE rn <= 100) AS mean_100d,
        AVG(cs) FILTER (WHERE rn <= 200) AS mean_200d,
        AVG(cs)                          AS mean_500d,
        STDDEV_SAMP(cs) FILTER (WHERE rn <= 5)   AS std_5d,
        STDDEV_SAMP(cs) FILTER (WHERE rn <= 10)  AS std_10d,
        STDDEV_SAMP(cs) FILTER (WHERE rn <= 20)  AS std_20d,
        STDDEV_SAMP(cs) FILTER (WHERE rn <= 50)  AS std_50d,
        STDDEV_SAMP(cs) FILTER (WHERE rn <= 100) AS std_100d,
        STDDEV_SAMP(cs) FILTER (WHERE rn <= 200) AS std_200d,
        STDDEV_SAMP(cs)                          AS std_500d,
        AVG(cs) FILTER (WHERE rn <= 5)   AS sff_5d_avg
    FROM (
        SELECT CASE WHEN (sub.foreign_sff * sub.institution_sff) > 0
                    THEN sub.foreign_sff + sub.institution_sff * w_inst
                    ELSE sub.foreign_sff END AS cs,
               ROW_NUMBER() OVER (ORDER BY sub.trade_date DESC) AS rn
        FROM mv_daily_sff sub, ref
        WHERE sub.stock_code = s.stock_code
          AND sub.trade_date <= ref.ref_date
        ORDER BY sub.trade_date DESC
        LIMIT 500
    ) t
) z;
$$ LANGUAGE sql STABLE;

-- 읽기 권한
GRANT EXECUTE ON FUNCTION fn_zscore_latest(float, date) TO korea_stock_reader;


-- =========================================================================
-- SQL 함수: fn_signals_latest — 최신(또는 지정) 날짜의 시그널 계산
-- MA 크로스오버, 수급 가속도, 외인-기관 동조율
--
-- 사용법:
--   SELECT * FROM fn_signals_latest(0.3);              -- 최신 날짜
--   SELECT * FROM fn_signals_latest(0.3, '2026-02-27'); -- 특정 날짜
-- =========================================================================

DROP FUNCTION IF EXISTS fn_signals_latest(float, date);

CREATE OR REPLACE FUNCTION fn_signals_latest(
    w_inst float DEFAULT 0.3,
    target_date date DEFAULT NULL
) RETURNS TABLE (
    stock_code varchar,
    ma_cross boolean,
    ma_diff double precision,
    acceleration double precision,
    sync_rate double precision
) AS $$
WITH ref AS (
    SELECT COALESCE(target_date, MAX(m.trade_date)) AS ref_date
    FROM mv_daily_sff m
)
SELECT s.stock_code,
       -- MA 골든크로스: 외국인 5일MA > 20일MA AND 전일은 반대
       (z.ma5_today > z.ma20_today AND z.ma5_prev <= z.ma20_prev) AS ma_cross,
       z.ma5_today - z.ma20_today AS ma_diff,
       -- 가속도: 최근 5일 combined_net 평균 / 직전 5일 평균
       CASE WHEN ABS(z.prev5_avg) < 1e-6 THEN NULL
            ELSE z.recent5_avg / z.prev5_avg END AS acceleration,
       -- 동조율: 최근 20일 중 외국인+기관 동시 매수 비율 (%)
       z.sync_rate * 100 AS sync_rate
FROM (
    SELECT DISTINCT stock_code FROM mv_daily_sff m, ref WHERE m.trade_date = ref.ref_date
) s
CROSS JOIN LATERAL (
    SELECT
        -- MA5/MA20 (외국인 순매수금액 기준)
        AVG(fa) FILTER (WHERE rn <= 5)  AS ma5_today,
        AVG(fa) FILTER (WHERE rn <= 20) AS ma20_today,
        AVG(fa) FILTER (WHERE rn >= 2 AND rn <= 6)  AS ma5_prev,
        AVG(fa) FILTER (WHERE rn >= 2 AND rn <= 21) AS ma20_prev,
        -- 가속도 (combined_net 기준)
        AVG(cn) FILTER (WHERE rn <= 5)  AS recent5_avg,
        AVG(cn) FILTER (WHERE rn >= 6 AND rn <= 10) AS prev5_avg,
        -- 동조율 (외국인>0 AND 기관>0 비율, 20일)
        AVG(CASE WHEN rn <= 20 AND fa > 0 AND ia > 0 THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE rn <= 20) AS sync_rate
    FROM (
        SELECT sub.foreign_net_amount AS fa,
               sub.institution_net_amount AS ia,
               -- combined_net: 같은 방향이면 합산, 아니면 외국인만
               CASE WHEN (sub.foreign_net_amount * sub.institution_net_amount) > 0
                    THEN sub.foreign_net_amount + sub.institution_net_amount * w_inst
                    ELSE sub.foreign_net_amount END AS cn,
               ROW_NUMBER() OVER (ORDER BY sub.trade_date DESC) AS rn
        FROM mv_daily_sff sub, ref
        WHERE sub.stock_code = s.stock_code
          AND sub.trade_date <= ref.ref_date
        ORDER BY sub.trade_date DESC
        LIMIT 21
    ) t
) z;
$$ LANGUAGE sql STABLE;

-- 읽기 권한
GRANT EXECUTE ON FUNCTION fn_signals_latest(float, date) TO korea_stock_reader;
