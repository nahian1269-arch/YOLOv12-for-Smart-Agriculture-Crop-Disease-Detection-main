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
    analysis jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.sensor_readings (
    id text primary key,
    project_id text,
    timestamp timestamptz not null,
    dht11_temp numeric,
    dht11_humidity numeric,
    dht7_temp numeric,
    mq5 numeric,
    mq7 numeric,
    mq135 numeric,
    lux numeric,
    rain_drop numeric,
    soil_moisture numeric,
    water_level numeric,
    motion integer
);

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

alter table public.users enable row level security;
alter table public.projects enable row level security;
alter table public.sensor_readings enable row level security;
alter table public.disease_detections enable row level security;
alter table public.weather_snapshots enable row level security;
alter table public.weather_predictions enable row level security;
alter table public.seasonal_analyses enable row level security;
alter table public.sensor_analyses enable row level security;

grant insert on public.users to anon;
grant insert on public.projects to anon;
grant insert on public.sensor_readings to anon;
grant insert on public.disease_detections to anon;
grant insert on public.weather_snapshots to anon;
grant insert on public.weather_predictions to anon;
grant insert on public.seasonal_analyses to anon;
grant insert on public.sensor_analyses to anon;

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
