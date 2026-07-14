create table if not exists public.users (
    id text primary key,
    name text not null,
    email text not null unique,
    phone text,
    status text not null default 'pending',
    role text not null default 'farmer',
    joined date,
    last_login timestamptz,
    location jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.projects (
    id text primary key,
    owner_id text not null,
    name text not null,
    area numeric not null,
    floors integer not null,
    lat double precision not null,
    lng double precision not null,
    goal text not null,
    weather_notes text,
    analysis jsonb,
    created_at timestamptz not null default now()
);

alter table public.projects add column if not exists weather_notes text;

create table if not exists public.sensor_readings (
    id text primary key,
    project_id text,
    timestamp timestamptz not null,
    dht11_temp numeric,
    dht11_humidity numeric,
    dht7_temp numeric,
    dht22_temp numeric,
    dht22_humidity numeric,
    mq2 numeric,
    mq3 numeric,
    mq5 numeric,
    mq7 numeric,
    mq135 numeric,
    lux numeric,
    ldr numeric,
    ph numeric,
    rain_drop numeric,
    soil_moisture numeric,
    water_level numeric,
    motion integer
);

alter table public.sensor_readings add column if not exists dht22_temp numeric;
alter table public.sensor_readings add column if not exists dht22_humidity numeric;
alter table public.sensor_readings add column if not exists mq2 numeric;
alter table public.sensor_readings add column if not exists mq3 numeric;
alter table public.sensor_readings add column if not exists ldr numeric;
alter table public.sensor_readings add column if not exists ph numeric;

create table if not exists public.disease_detections (
    id text primary key,
    user_id text,
    project_id text,
    timestamp timestamptz not null,
    image text,
    summary jsonb not null default '{}'::jsonb,
    detections jsonb not null default '[]'::jsonb,
    recommendations jsonb not null default '[]'::jsonb
);

alter table public.disease_detections add column if not exists user_id text;
alter table public.disease_detections add column if not exists detections jsonb not null default '[]'::jsonb;

create table if not exists public.sensor_analyses (
    id text primary key,
    project_id text,
    generated_at timestamptz not null,
    next_analysis_at timestamptz not null,
    window_start timestamptz,
    window_end timestamptz,
    sample_count integer not null,
    health_score numeric not null,
    risk_level text not null,
    averages jsonb not null default '{}'::jsonb,
    trends jsonb not null default '{}'::jsonb,
    anomalies jsonb not null default '[]'::jsonb,
    feedback jsonb not null default '[]'::jsonb,
    model_name text not null
);

create table if not exists public.weather_snapshots (
    id text primary key,
    user_id text not null,
    project_id text,
    observed_at timestamptz not null,
    weather_date date not null,
    latitude double precision not null,
    longitude double precision not null,
    timezone text,
    current_weather jsonb not null,
    daily_weather jsonb not null,
    created_at timestamptz not null default now(),
    unique (user_id, project_id, weather_date)
);

create table if not exists public.weather_predictions (
    id text primary key,
    user_id text not null,
    project_id text,
    predicted_at timestamptz not null,
    target_at timestamptz not null,
    latitude double precision not null,
    longitude double precision not null,
    model_name text not null,
    prediction jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists public.seasonal_analyses (
    id text primary key,
    user_id text not null,
    project_id text,
    generated_at timestamptz not null,
    next_analysis_at timestamptz not null,
    latitude double precision not null,
    longitude double precision not null,
    preferred_plants jsonb not null default '[]'::jsonb,
    floor_recommendations jsonb not null default '[]'::jsonb,
    source text not null,
    report text not null,
    weather_summary jsonb not null default '{}'::jsonb
);

create table if not exists public.activities (
    id text primary key,
    timestamp timestamptz not null,
    actor_type text not null,
    actor_id text not null,
    action text not null,
    details text not null default ''
);

create table if not exists public.device_commands (
    id text primary key,
    device text not null,
    value text not null,
    status text not null default 'queued',
    created_at timestamptz not null default now(),
    created_by text,
    acknowledged_at timestamptz,
    esp_response text not null default ''
);

create table if not exists public.notifications (
    id text primary key,
    user_id text not null,
    project_id text,
    level text not null,
    title text not null,
    message text not null,
    source text not null default 'system',
    read boolean not null default false,
    created_at timestamptz not null default now()
);

create table if not exists public.iot_devices (
    id text primary key,
    role text not null,
    status text not null default 'online',
    ip text,
    rssi numeric,
    last_seen timestamptz not null,
    extra jsonb not null default '{}'::jsonb
);

alter table public.users enable row level security;
alter table public.projects enable row level security;
alter table public.sensor_readings enable row level security;
alter table public.disease_detections enable row level security;
alter table public.weather_snapshots enable row level security;
alter table public.weather_predictions enable row level security;
alter table public.seasonal_analyses enable row level security;
alter table public.sensor_analyses enable row level security;
alter table public.activities enable row level security;
alter table public.device_commands enable row level security;
alter table public.notifications enable row level security;
alter table public.iot_devices enable row level security;

grant insert on public.users to anon;
grant insert on public.projects to anon;
grant insert on public.sensor_readings to anon;
grant insert on public.disease_detections to anon;
grant insert on public.weather_snapshots to anon;
grant insert on public.weather_predictions to anon;
grant insert on public.seasonal_analyses to anon;
grant insert on public.sensor_analyses to anon;
grant insert on public.activities to anon;
grant insert on public.device_commands to anon;
grant insert on public.notifications to anon;
grant insert on public.iot_devices to anon;

grant select, update on public.users to anon;
grant select, update on public.projects to anon;
grant select, update on public.sensor_readings to anon;
grant select, update on public.disease_detections to anon;
grant select, update on public.weather_snapshots to anon;
grant select, update on public.weather_predictions to anon;
grant select, update on public.seasonal_analyses to anon;
grant select, update on public.sensor_analyses to anon;
grant select, update on public.activities to anon;
grant select, update on public.device_commands to anon;
grant select, update on public.notifications to anon;
grant select, update on public.iot_devices to anon;

drop policy if exists "NuroAgro local app user inserts" on public.users;
create policy "NuroAgro local app user inserts"
on public.users for insert to anon with check (true);

drop policy if exists "NuroAgro local app project inserts" on public.projects;
create policy "NuroAgro local app project inserts"
on public.projects for insert to anon with check (true);

drop policy if exists "NuroAgro local app sensor inserts" on public.sensor_readings;
create policy "NuroAgro local app sensor inserts"
on public.sensor_readings for insert to anon with check (true);

drop policy if exists "NuroAgro local app disease inserts" on public.disease_detections;
create policy "NuroAgro local app disease inserts"
on public.disease_detections for insert to anon with check (true);

drop policy if exists "NuroAgro local app weather snapshot inserts" on public.weather_snapshots;
create policy "NuroAgro local app weather snapshot inserts"
on public.weather_snapshots for insert to anon with check (true);

drop policy if exists "NuroAgro local app weather prediction inserts" on public.weather_predictions;
create policy "NuroAgro local app weather prediction inserts"
on public.weather_predictions for insert to anon with check (true);

drop policy if exists "NuroAgro local app seasonal analysis inserts" on public.seasonal_analyses;
create policy "NuroAgro local app seasonal analysis inserts"
on public.seasonal_analyses for insert to anon with check (true);

drop policy if exists "NuroAgro local app sensor analysis inserts" on public.sensor_analyses;
create policy "NuroAgro local app sensor analysis inserts"
on public.sensor_analyses for insert to anon with check (true);

drop policy if exists "NuroAgro local app activity inserts" on public.activities;
create policy "NuroAgro local app activity inserts"
on public.activities for insert to anon with check (true);

drop policy if exists "NuroAgro local app device command inserts" on public.device_commands;
create policy "NuroAgro local app device command inserts"
on public.device_commands for insert to anon with check (true);

drop policy if exists "NuroAgro local app notification inserts" on public.notifications;
create policy "NuroAgro local app notification inserts"
on public.notifications for insert to anon with check (true);

drop policy if exists "NuroAgro local app iot device inserts" on public.iot_devices;
create policy "NuroAgro local app iot device inserts"
on public.iot_devices for insert to anon with check (true);

drop policy if exists "NuroAgro local app user reads" on public.users;
create policy "NuroAgro local app user reads"
on public.users for select to anon using (true);

drop policy if exists "NuroAgro local app project reads" on public.projects;
create policy "NuroAgro local app project reads"
on public.projects for select to anon using (true);

drop policy if exists "NuroAgro local app sensor reads" on public.sensor_readings;
create policy "NuroAgro local app sensor reads"
on public.sensor_readings for select to anon using (true);

drop policy if exists "NuroAgro local app disease reads" on public.disease_detections;
create policy "NuroAgro local app disease reads"
on public.disease_detections for select to anon using (true);

drop policy if exists "NuroAgro local app weather snapshot reads" on public.weather_snapshots;
create policy "NuroAgro local app weather snapshot reads"
on public.weather_snapshots for select to anon using (true);

drop policy if exists "NuroAgro local app weather prediction reads" on public.weather_predictions;
create policy "NuroAgro local app weather prediction reads"
on public.weather_predictions for select to anon using (true);

drop policy if exists "NuroAgro local app seasonal analysis reads" on public.seasonal_analyses;
create policy "NuroAgro local app seasonal analysis reads"
on public.seasonal_analyses for select to anon using (true);

drop policy if exists "NuroAgro local app sensor analysis reads" on public.sensor_analyses;
create policy "NuroAgro local app sensor analysis reads"
on public.sensor_analyses for select to anon using (true);

drop policy if exists "NuroAgro local app activity reads" on public.activities;
create policy "NuroAgro local app activity reads"
on public.activities for select to anon using (true);

drop policy if exists "NuroAgro local app device command reads" on public.device_commands;
create policy "NuroAgro local app device command reads"
on public.device_commands for select to anon using (true);

drop policy if exists "NuroAgro local app notification reads" on public.notifications;
create policy "NuroAgro local app notification reads"
on public.notifications for select to anon using (true);

drop policy if exists "NuroAgro local app iot device reads" on public.iot_devices;
create policy "NuroAgro local app iot device reads"
on public.iot_devices for select to anon using (true);

drop policy if exists "NuroAgro local app user updates" on public.users;
create policy "NuroAgro local app user updates"
on public.users for update to anon using (true) with check (true);

drop policy if exists "NuroAgro local app device command updates" on public.device_commands;
create policy "NuroAgro local app device command updates"
on public.device_commands for update to anon using (true) with check (true);

drop policy if exists "NuroAgro local app notification updates" on public.notifications;
create policy "NuroAgro local app notification updates"
on public.notifications for update to anon using (true) with check (true);

drop policy if exists "NuroAgro local app iot device updates" on public.iot_devices;
create policy "NuroAgro local app iot device updates"
on public.iot_devices for update to anon using (true) with check (true);










