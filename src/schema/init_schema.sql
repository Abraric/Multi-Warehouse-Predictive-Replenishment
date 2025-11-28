-- Multi-Warehouse Predictive Replenishment Database Schema

-- Warehouses table
CREATE TABLE IF NOT EXISTS warehouses (
    warehouse_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    capacity_volume_m3 DECIMAL(12, 2),
    capacity_weight_kg DECIMAL(12, 2),
    lead_time_days INTEGER,
    transfer_cost_per_km DECIMAL(10, 4),
    holding_cost_per_unit_per_day DECIMAL(10, 4),
    stockout_cost_per_unit DECIMAL(10, 2),
    num_docks INTEGER,
    dock_window_start TIME,
    dock_window_end TIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SKUs table
CREATE TABLE IF NOT EXISTS skus (
    sku_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    volume_m3_per_unit DECIMAL(10, 6),
    weight_kg_per_unit DECIMAL(10, 4),
    unit_cost DECIMAL(10, 2),
    perishability_ttl_days INTEGER,
    requires_refrigerated BOOLEAN DEFAULT FALSE,
    safety_stock_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inventory snapshots table
CREATE TABLE IF NOT EXISTS inventory_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    warehouse_id VARCHAR(50) NOT NULL,
    sku_id VARCHAR(50) NOT NULL,
    quantity_on_hand INTEGER NOT NULL DEFAULT 0,
    quantity_reserved INTEGER NOT NULL DEFAULT 0,
    quantity_available INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id),
    UNIQUE(snapshot_date, warehouse_id, sku_id)
);

-- Sales history table
CREATE TABLE IF NOT EXISTS sales_history (
    sale_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    warehouse_id VARCHAR(50) NOT NULL,
    sku_id VARCHAR(50) NOT NULL,
    quantity_sold INTEGER NOT NULL,
    revenue DECIMAL(12, 2),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id),
    INDEX idx_sales_date (date),
    INDEX idx_sales_wh_sku (warehouse_id, sku_id)
);

-- Inbound purchase orders table
CREATE TABLE IF NOT EXISTS inbound_pos (
    po_id VARCHAR(100) PRIMARY KEY,
    warehouse_id VARCHAR(50) NOT NULL,
    sku_id VARCHAR(50) NOT NULL,
    order_date DATE NOT NULL,
    expected_arrival_date DATE,
    quantity_ordered INTEGER NOT NULL,
    quantity_received INTEGER,
    supplier_lead_time_days INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id)
);

-- Fleet table
CREATE TABLE IF NOT EXISTS fleet (
    truck_id VARCHAR(50) PRIMARY KEY,
    truck_type VARCHAR(50) NOT NULL,
    volume_capacity_m3 DECIMAL(10, 2),
    weight_capacity_kg DECIMAL(10, 2),
    cost_per_km DECIMAL(10, 4),
    is_refrigerated BOOLEAN DEFAULT FALSE,
    max_daily_hours DECIMAL(4, 2),
    fixed_cost_per_trip DECIMAL(10, 2)
);

-- Drivers table
CREATE TABLE IF NOT EXISTS drivers (
    driver_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    shift_start TIME,
    shift_end TIME,
    max_hours_per_day DECIMAL(4, 2),
    can_drive_refrigerated BOOLEAN DEFAULT FALSE
);

-- Transfers table (executed transfers)
CREATE TABLE IF NOT EXISTS transfers (
    transfer_id SERIAL PRIMARY KEY,
    plan_id VARCHAR(100),
    from_warehouse_id VARCHAR(50) NOT NULL,
    to_warehouse_id VARCHAR(50) NOT NULL,
    sku_id VARCHAR(50) NOT NULL,
    quantity INTEGER NOT NULL,
    ship_date DATE NOT NULL,
    expected_arrival_date DATE,
    actual_arrival_date DATE,
    truck_id VARCHAR(50),
    driver_id VARCHAR(50),
    status VARCHAR(50) DEFAULT 'planned',
    cost DECIMAL(12, 2),
    FOREIGN KEY (from_warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (to_warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id),
    FOREIGN KEY (truck_id) REFERENCES fleet(truck_id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
);

-- Replenishment plans table
CREATE TABLE IF NOT EXISTS replenishment_plans (
    plan_id VARCHAR(100) PRIMARY KEY,
    run_date DATE NOT NULL,
    from_wh VARCHAR(50),
    to_wh VARCHAR(50),
    sku_id VARCHAR(50) NOT NULL,
    qty INTEGER NOT NULL,
    ship_date DATE NOT NULL,
    expected_arrival DATE,
    truck_type VARCHAR(50),
    driver_id VARCHAR(50),
    estimated_cost DECIMAL(12, 2),
    reason_code VARCHAR(100),
    explanation_text TEXT,
    delta_cost_if_not_transferred DECIMAL(12, 2),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id)
);

-- Cost logs table
CREATE TABLE IF NOT EXISTS cost_logs (
    log_id SERIAL PRIMARY KEY,
    plan_id VARCHAR(100),
    date DATE NOT NULL,
    cost_type VARCHAR(50) NOT NULL,
    cost_amount DECIMAL(12, 2) NOT NULL,
    description TEXT,
    FOREIGN KEY (plan_id) REFERENCES replenishment_plans(plan_id)
);

