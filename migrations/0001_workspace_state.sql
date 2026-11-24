-- Sypra durable workspace state migration.
--
-- The application applies this schema automatically for SQLite and PostgreSQL.
-- The JSON payload is intentional during the compatibility phase: the Store
-- repository preserves its atomic all-or-nothing mutation semantics while the
-- domain tables continue to evolve. PostgreSQL deployments must use one Sypra
-- API writer until the planned normalized shared-workspace schema is released.
CREATE TABLE IF NOT EXISTS sypra_workspace_state (
    id BIGINT PRIMARY KEY,
    payload BYTEA NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
