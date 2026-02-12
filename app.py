import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard Master: Cost vs Revenue", page_icon="💰", layout="wide")

# ==========================================
# ⚙️ TUS URLS Y DATOS
# ==========================================
SHEET_ID = "1WuBv1esTxZAfC07BPwWzjsz5TZqfUHa6MOzNIAEOMew"
GID_ANDROID = "368085162"    # Datos por AOS/dia
GID_IOS = "1225911759"       # Datos por IOS/dia

def get_url(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

# --- FUNCIÓN DE BÚSQUEDA EXACTA (BLINDADA) ---
def find_column_strict(columns, candidates):
    """Busca columnas de forma segura para evitar errores de lectura."""
    cols_clean = [str(c).lower().strip() for c in columns]
    
    for cand in candidates:
        cand = cand.lower().strip()
        if cand in cols_clean:
            return columns[cols_clean.index(cand)]
        if len(cand) > 3:
            for i, col in enumerate(cols_clean):
                if cand in col:
                    return columns[i]
    return None

@st.cache_data(ttl=600)
def load_data():
    urls = {'Android': get_url(GID_ANDROID), 'iOS': get_url(GID_IOS)}
    dfs = []
    
    for os_name, url in urls.items():
        try:
            # 1. Leer CSV
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip().str.replace('"', '')
            
            # 2. MAPEO DE COLUMNAS
            col_mapping = {}
            
            # Columnas Básicas
            c_date = find_column_strict(df.columns, ['date', 'day', 'fecha', 'v', 'time'])
            c_country = find_column_strict(df.columns, ['country', 'geo', 'geo/os', 'pais'])
            c_cost = find_column_strict(df.columns, ['cost', 'coste', 'spend', 'total cost'])
            c_rev = find_column_strict(df.columns, ['revenue total', 'revenue', 'ingresos', 'gain'])

            # Columnas ZP y PC
            c_rec_vis = find_column_strict(df.columns, ['received visits zp', 'received visits'])
            c_sold_vis = find_column_strict(df.columns, ['sold visits zp', 'sold visits'])
            c_perc_sold = find_column_strict(df.columns, ['%sold zp', '% sold zp', 'sold %'])
            c_cpm_zp = find_column_strict(df.columns, ['cpm zp', 'cpm'])
            c_cpm_pc = find_column_strict(df.columns, ['cpm pc'])

            # Asignamos nombres estándar
            if c_date: col_mapping[c_date] = 'Date'
            if c_country: col_mapping[c_country] = 'Country'
            if c_cost: col_mapping[c_cost] = 'Cost'
            if c_rev: col_mapping[c_rev] = 'Revenue'
            
            if c_rec_vis: col_mapping[c_rec_vis] = 'Received Visits ZP'
            if c_sold_vis: col_mapping[c_sold_vis] = 'Sold Visits ZP'
            if c_perc_sold: col_mapping[c_perc_sold] = '% Sold ZP'
            if c_cpm_zp: col_mapping[c_cpm_zp] = 'CPM ZP'
            if c_cpm_pc: col_mapping[c_cpm_pc] = 'CPM PC'

            if col_mapping:
                df.rename(columns=col_mapping, inplace=True)
            
            # 3. FILTRO DE SEGURIDAD (FECHAS)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']) 
            else:
                continue

            # 4. LIMPIEZA NUMÉRICA
            numeric_cols = ['Cost', 'Revenue', 'Received Visits ZP', 'Sold Visits ZP', '% Sold ZP', 'CPM ZP', 'CPM PC']
            for col in numeric_cols:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace('$', '', regex=False)\
                                                     .str.replace(',', '', regex=False)\
                                                     .str.replace('%', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['OS'] = os_name
            dfs.append(df)
            
        except Exception as e:
            st.error(f"Error procesando {os_name}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True).fillna(0)
    return None

# --- CARGAR DATOS ---
df = load_data()

# --- INTERFAZ ---
st.title("📊 Dashboard Financiero & Métricas ZP")

if df is not None and not df.empty:
    
    # --- FILTROS ---
    st.sidebar.header("Filtros Globales")
    
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    if pd.isnull(min_date) or pd.isnull(max_date):
        st.error("⚠️ No hay fechas válidas. Revisa el Excel.")
        st.stop()
        
    date_range = st.sidebar.date_input("Rango de Fechas", [min_date, max_date])
    
    selected_os = st.sidebar.multiselect("Sistema Operativo", df['OS'].unique(), default=df['OS'].unique())
    
    countries_list = sorted(df['Country'].unique().astype(str)) if 'Country' in df.columns else []
    selected_countries = st.sidebar.multiselect("Países", countries_list, default=countries_list)

    # --- APLICAR FILTROS ---
    mask = (df['OS'].isin(selected_os))
    if len(date_range) == 2:
        mask = mask & (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
    if selected_countries and 'Country' in df.columns:
        mask = mask & (df['Country'].isin(selected_countries))
        
    df_filtered = df[mask]

    # --- KPI SUMMARY ---
    st.subheader("💰 Resumen Financiero")
    
    cost = df_filtered['Cost'].sum() if 'Cost' in df_filtered.columns else 0
    rev = df_filtered['Revenue'].sum() if 'Revenue' in df_filtered.columns else 0
    profit = rev - cost
    roi = (profit / cost * 100) if cost > 0 else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Coste Total", f"${cost:,.2f}")
    k2.metric("Revenue Total", f"${rev:,.2f}")
    k3.metric("Beneficio", f"${profit:,.2f}")
    k4.metric("ROI Global", f"{roi:.2f}%")
    
    st.divider()

    # ========================================================
    # 1️⃣ GRÁFICA PRINCIPAL: COSTE VS REVENUE DIARIO
    # ========================================================
    st.subheader("📅 1. Evolución Financiera: Coste vs Revenue")
    
    if 'Cost' in df_filtered.columns and 'Revenue' in df_filtered.columns:
        df_daily = df_filtered.groupby('Date')[['Cost', 'Revenue']].sum().reset_index()
        fig_daily = px.line(
            df_daily.melt(id_vars='Date'), 
            x='Date', 
            y='value', 
            color='variable',
            color_discrete_map={'Cost':'#EF553B', 'Revenue':'#00CC96'},
            markers=True,
            title="Comparativa Diaria de Ingresos y Gastos"
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        st.warning("No hay datos de Coste o Revenue disponibles.")

    st.divider()

    # ========================================================
    # 2️⃣ GRÁFICA SECUNDARIA: COMPARATIVA CPM (ZP vs PC)
    # ========================================================
    st.subheader("📉 2. Calidad del Tráfico: CPM ZP vs CPM PC")
    
    cpm_cols = ['CPM ZP', 'CPM PC']
    existing_cpm = [c for c in cpm_cols if c in df_filtered.columns]
    
    if existing_cpm:
        # Promedio diario
        df_cpm = df_filtered.groupby('Date')[existing_cpm].mean().reset_index()
        
        fig_cpm = px.line(
            df_cpm.melt(id_vars='Date', var_name='Tipo', value_name='CPM ($)'),
            x='Date', 
            y='CPM ($)', 
            color='Tipo',
            markers=True,
            color_discrete_map={'CPM ZP': '#FFA15A', 'CPM PC': '#636EFA'},
            title="Evolución del Coste por Mil (Promedio Diario)"
        )
        st.plotly_chart(fig_cpm, use_container_width=True)
    else:
        st.warning("No se encontraron columnas de CPM en los datos.")

    st.divider()

    # ========================================================
    # 3️⃣ GRÁFICA TERCIARIA: MÉTRICAS ZP (SELECTOR)
    # ========================================================
    st.subheader("📈 3. Análisis Detallado ZP (Personalizable)")
    
    possible_zp = ['Received Visits ZP', 'Sold Visits ZP', '% Sold ZP', 'CPM ZP']
    available_zp = [c for c in possible_zp if c in df_filtered.columns]
    
    col_sel, col_chart = st.columns([1, 3])
    
    with col_sel:
        st.info("👇 Configura esta gráfica:")
        selected_metrics = st.multiselect(
            "Selecciona Métricas:", 
            options=available_zp,
            default=available_zp[:2] if available_zp else None
        )
    
    with col_chart:
        if selected_metrics:
            agg_rules = {}
            for m in selected_metrics:
                if 'Visits' in m:
                    agg_rules[m] = 'sum'
                else:
                    agg_rules[m] = 'mean'
            
            df_zp = df_filtered.groupby('Date')[selected_metrics].agg(agg_rules).reset_index()
            fig_zp = px.line(
                df_zp.melt(id_vars='Date', var_name='Metric', value_name='Value'), 
                x='Date', y='Value', color='Metric', markers=True,
                title="Tendencia de Métricas Seleccionadas"
            )
            st.plotly_chart(fig_zp, use_container_width=True)
        else:
            st.warning("Selecciona al menos una métrica en el menú izquierdo.")

    # --- FOOTER ---
    with st.expander("📂 Ver Datos Brutos"):
        st.dataframe(df_filtered)

else:
    st.info("⏳ Cargando datos... (Si tarda, revisa permisos del Sheet)")
