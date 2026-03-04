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
GID_TARGETS = "0"            # Datos pestaña Zeropark

def get_url(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

# --- FUNCIÓN DE BÚSQUEDA EXACTA (BLINDADA) ---
def find_column_strict(columns, candidates):
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

# --- FUNCIÓN LIMPIEZA DE TARGETS (REUTILIZABLE Y SEGURA) ---
def clean_target_df(df):
    """Limpia cualquier dataframe con la lógica de Zeropark"""
    original_cols = list(df.columns)
    col_mapping = {}
    
    c_date = find_column_strict(df.columns, ['date', 'day', 'fecha', 'v', 'time', 'interval', 'period'])
    c_target = find_column_strict(df.columns, ['target', 'id target', 'subid', 'publisher', 'source', 'placement', 'keyword', 'campaign'])
    
    if c_date: col_mapping[c_date] = 'Date'
    if c_target: col_mapping[c_target] = 'Target'
    
    if col_mapping:
        df.rename(columns=col_mapping, inplace=True)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']) 

    # ✂️ EXTRAER GEO Y OS DEL TARGET (AHORA CON PROTECCIÓN N/A)
    if 'Target' in df.columns:
        # Extraemos las letras tras el guión. Si no hay guión, rellenamos con 'N/A' para que la columna no se borre.
        df['GEO'] = df['Target'].astype(str).str.extract(r'-(.{2})', expand=False).str.upper().fillna('N/A')
        # OS: iOS si tiene "io", si no Android
        df['OS'] = df['Target'].astype(str).apply(lambda x: 'iOS' if 'io' in x.lower() else 'Android')
    
    for col in df.columns:
        if col not in ['Date', 'Target', 'GEO', 'OS'] and df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                pass
    
    df.attrs['original_cols'] = original_cols
    return df

# --- MOTOR DE INTERFAZ PARA TARGETS (PLANTILLA INDEPENDIENTE) ---
def render_targets_ui(df_targets, key_prefix):
    """Genera los filtros, gráficas y tabla. 100% independiente por pestaña."""
    col_f1, col_f2, col_f3 = st.columns(3)
    min_t_date = df_targets['Date'].min()
    max_t_date = df_targets['Date'].max()
    
    with col_f1:
        t_date_range = st.date_input("📅 Filtrar Fechas:", [min_t_date, max_t_date], key=f"{key_prefix}_dates")
        
    with col_f2:
        all_geos = sorted(df_targets['GEO'].astype(str).unique()) if 'GEO' in df_targets.columns else []
        geo_selected = st.multiselect("🌍 Filtrar GEO:", options=all_geos, placeholder="Selecciona un país...", key=f"{key_prefix}_geo")

    with col_f3:
        all_os = sorted(df_targets['OS'].astype(str).unique()) if 'OS' in df_targets.columns else []
        os_selected = st.multiselect("📱 Filtrar OS:", options=all_os, placeholder="Selecciona un sistema...", key=f"{key_prefix}_os")

    all_targets = sorted(df_targets['Target'].astype(str).unique())
    t_selected = st.multiselect("🔍 Buscar/Filtrar Target específico (Opcional):", options=all_targets, placeholder="Selecciona uno o varios targets...", key=f"{key_prefix}_target")

    mask_t = pd.Series(True, index=df_targets.index)
    
    if len(t_date_range) == 2:
        mask_t = mask_t & (df_targets['Date'] >= pd.to_datetime(t_date_range[0])) & (df_targets['Date'] <= pd.to_datetime(t_date_range[1]))
    if geo_selected:
        mask_t = mask_t & (df_targets['GEO'].astype(str).isin(geo_selected))
    if os_selected:
        mask_t = mask_t & (df_targets['OS'].astype(str).isin(os_selected))
    if t_selected:
        mask_t = mask_t & (df_targets['Target'].astype(str).isin(t_selected))
        
    df_t_filtered = df_targets[mask_t]

    numeric_columns = df_t_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    group_cols = ['Date']
    if 'GEO' in df_t_filtered.columns:
        group_cols.append('GEO')
    if 'OS' in df_t_filtered.columns:
        group_cols.append('OS')
    group_cols.append('Target')
        
    # Usamos dropna=False por si algún dato es extraño, así no borra la fila
    df_grouped = df_t_filtered.groupby(group_cols, dropna=False)[numeric_columns].sum().reset_index()
    
    if geo_selected and os_selected:
        st.divider()
        st.subheader("📈 Rendimiento Diario de Targets Filtrados")
        
        col_received = next((c for c in numeric_columns if 'received' in c.lower() or 'visit' in c.lower()), None)
        col_cpm = next((c for c in numeric_columns if 'cpm' in c.lower()), None)
        
        if col_received:
            df_chart_received = df_t_filtered.groupby(['Date', 'Target'], dropna=False)[col_received].sum().reset_index()
            fig_received = px.line(
                df_chart_received, x='Date', y=col_received, color='Target', markers=True,
                title=f"Evolución de {col_received}"
            )
            st.plotly_chart(fig_received, use_container_width=True) 
        else:
            st.warning("No se encontró columna para Received Traffic/Visits.")
        
        st.write("")
        
        if col_cpm:
            df_chart_cpm = df_t_filtered.groupby(['Date', 'Target'], dropna=False)[col_cpm].mean().reset_index()
            fig_cpm = px.line(
                df_chart_cpm, x='Date', y=col_cpm, color='Target', markers=True,
                title=f"Evolución de {col_cpm} (Promedio)"
            )
            st.plotly_chart(fig_cpm, use_container_width=True)
        else:
            st.warning("No se encontró columna para CPM.")
            
        st.divider()
    else:
        st.info("💡 **Tip:** Selecciona un GEO y un OS en los filtros de arriba para desbloquear las gráficas de rendimiento visual.")

    df_grouped['Date'] = pd.to_datetime(df_grouped['Date']).dt.strftime('%Y-%m-%d')
    sort_ascending = [False] + [True] * (len(group_cols) - 1)
    df_grouped = df_grouped.sort_values(by=group_cols, ascending=sort_ascending)
    
    st.markdown(f"**Total de filas mostradas:** {len(df_grouped)}")
    st.dataframe(df_grouped, use_container_width=True, height=600, hide_index=True)


# --- CARGA DEL DASHBOARD PRINCIPAL (AOS/IOS) ---
@st.cache_data(ttl=600)
def load_data():
    urls = {'Android': get_url(GID_ANDROID), 'iOS': get_url(GID_IOS)}
    dfs = []
    for os_name, url in urls.items():
        try:
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip().str.replace('"', '')
            
            col_mapping = {}
            c_date = find_column_strict(df.columns, ['date', 'day', 'fecha', 'v', 'time'])
            c_country = find_column_strict(df.columns, ['country', 'geo', 'geo/os', 'pais'])
            c_cost = find_column_strict(df.columns, ['cost', 'coste', 'spend', 'total cost'])
            c_rev = find_column_strict(df.columns, ['revenue total', 'revenue', 'ingresos', 'gain'])
            c_rec_vis = find_column_strict(df.columns, ['received visits zp', 'received visits'])
            c_sold_vis = find_column_strict(df.columns, ['sold visits zp', 'sold visits'])
            c_perc_sold = find_column_strict(df.columns, ['%sold zp', '% sold zp', 'sold %'])
            c_cpm_zp = find_column_strict(df.columns, ['cpm zp', 'cpm'])
            c_cpm_pc = find_column_strict(df.columns, ['cpm pc'])

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
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']) 
            else:
                continue

            numeric_cols = ['Cost', 'Revenue', 'Received Visits ZP', 'Sold Visits ZP', '% Sold ZP', 'CPM ZP', 'CPM PC']
            for col in numeric_cols:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace('%', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            if 'Cost' in df.columns and 'Revenue' in df.columns:
                df['Gain'] = df['Revenue'] - df['Cost']
            else:
                df['Gain'] = 0

            df['OS'] = os_name
            dfs.append(df)
            
        except Exception as e:
            st.error(f"Error procesando {os_name}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True).fillna(0)
    return None

# --- CARGA DE LA PESTAÑA TARGETS DESDE GOOGLE SHEETS ---
@st.cache_data(ttl=600)
def load_targets_data():
    url = get_url(GID_TARGETS)
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.replace('"', '')
        return clean_target_df(df)
    except Exception as e:
        st.error(f"Error cargando pestaña Zeropark: {e}")
        return None

# --- CARGAR AMBAS BASES DE DATOS NATIVAS ---
df_main = load_data()
df_targets = load_targets_data()

# --- INTERFAZ ---
st.title("📊 Dashboard Financiero & Operativo")

# 🌟 AQUÍ SE CREAN LAS TRES PESTAÑAS
tab_principal, tab_targets, tab_csv = st.tabs([
    "📈 Dashboard Principal", 
    "🎯 Análisis de Targets (Google Sheets)", 
    "📁 Analizar CSV Local (Independiente)"
])

# ==============================================================================
# 🗂️ PESTAÑA 1: DASHBOARD PRINCIPAL
# ==============================================================================
with tab_principal:
    if df_main is not None and not df_main.empty:
        
        st.sidebar.header("Filtros Globales (Dashboard Principal)")
        min_date = df_main['Date'].min()
        max_date = df_main['Date'].max()
        
        if pd.isnull(min_date) or pd.isnull(max_date):
            st.error("⚠️ No hay fechas válidas. Revisa el Excel.")
            st.stop()
            
        date_range = st.sidebar.date_input("Rango de Fechas", [min_date, max_date], key="dash_dates")
        selected_os = st.sidebar.multiselect("Sistema Operativo", df_main['OS'].unique(), default=df_main['OS'].unique(), key="dash_os")
        countries_list = sorted(df_main['Country'].unique().astype(str)) if 'Country' in df_main.columns else []
        selected_countries = st.sidebar.multiselect("Países", countries_list, default=countries_list, key="dash_countries")

        mask = (df_main['OS'].isin(selected_os))
        if len(date_range) == 2:
            mask = mask & (df_main['Date'] >= pd.to_datetime(date_range[0])) & (df_main['Date'] <= pd.to_datetime(date_range[1]))
        if selected_countries and 'Country' in df_main.columns:
            mask = mask & (df_main['Country'].isin(selected_countries))
            
        df_filtered = df_main[mask]

        st.subheader("💰 Resumen Financiero")
        cost = df_filtered['Cost'].sum() if 'Cost' in df_filtered.columns else 0
        rev = df_filtered['Revenue'].sum() if 'Revenue' in df_filtered.columns else 0
        profit = df_filtered['Gain'].sum() if 'Gain' in df_filtered.columns else (rev - cost)
        roi = (profit / cost * 100) if cost > 0 else 0
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Coste Total", f"${cost:,.2f}")
        k2.metric("Revenue Total", f"${rev:,.2f}")
        k3.metric("Beneficio", f"${profit:,.2f}")
        k4.metric("ROI Global", f"{roi:.2f}%")
        
        st.divider()

        st.subheader("📅 1. Evolución Financiera: Coste, Revenue y Gain")
        if 'Cost' in df_filtered.columns and 'Revenue' in df_filtered.columns:
            df_daily = df_filtered.groupby('Date')[['Cost', 'Revenue', 'Gain']].sum().reset_index()
            fig_daily = px.line(
                df_daily.melt(id_vars='Date', value_vars=['Cost', 'Revenue', 'Gain']), 
                x='Date', y='value', color='variable',
                color_discrete_map={'Cost':'#EF553B', 'Revenue':'#00CC96', 'Gain':'#2962FF'},
                markers=True
            )
            st.plotly_chart(fig_daily, use_container_width=True)

        st.divider()

        st.subheader("📉 2. Calidad del Tráfico: CPM ZP vs CPM PC")
        cpm_cols = ['CPM ZP', 'CPM PC']
        existing_cpm = [c for c in cpm_cols if c in df_filtered.columns]
        if existing_cpm:
            df_cpm = df_filtered.groupby('Date')[existing_cpm].mean().reset_index()
            fig_cpm = px.line(
                df_cpm.melt(id_vars='Date', var_name='Tipo', value_name='CPM ($)'),
                x='Date', y='CPM ($)', color='Tipo', markers=True,
                color_discrete_map={'CPM ZP': '#FFA15A', 'CPM PC': '#636EFA'}
            )
            st.plotly_chart(fig_cpm, use_container_width=True)

        st.divider()

        st.subheader("📈 3. Análisis Detallado ZP (Personalizable)")
        possible_zp = ['Received Visits ZP', 'Sold Visits ZP', '% Sold ZP', 'CPM ZP']
        available_zp = [c for c in possible_zp if c in df_filtered.columns]
        
        col_sel, col_chart = st.columns([1, 3])
        with col_sel:
            selected_metrics = st.multiselect("Selecciona Métricas:", options=available_zp, default=available_zp[:2] if available_zp else None, key="dash_metrics")
        with col_chart:
            if selected_metrics:
                agg_rules = {m: 'sum' if 'Visits' in m else 'mean' for m in selected_metrics}
                df_zp = df_filtered.groupby('Date')[selected_metrics].agg(agg_rules).reset_index()
                fig_zp = px.line(
                    df_zp.melt(id_vars='Date', var_name='Metric', value_name='Value'), 
                    x='Date', y='Value', color='Metric', markers=True
                )
                st.plotly_chart(fig_zp, use_container_width=True)

        st.divider()

        st.subheader("📂 Datos Detallados")
        df_display = df_filtered.copy()
        if 'Date' in df_display.columns:
            df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
            df_display = df_display.sort_values(by='Date', ascending=False)
        st.markdown(f"**Total de registros mostrados:** {len(df_display)}")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    else:
        st.info("⏳ Cargando datos del Dashboard...")

# ==============================================================================
# 🎯 PESTAÑA 2: ANÁLISIS DE TARGETS (ZEROPARK GOOGLE SHEETS)
# ==============================================================================
with tab_targets:
    st.subheader("🎯 Rendimiento desde Google Sheets (Pestaña 2)")
    
    if df_targets is not None and not df_targets.empty:
        if 'Date' in df_targets.columns and 'Target' in df_targets.columns:
            # Filtros con llave "zp"
            render_targets_ui(df_targets, "zp")
        else:
            cols_found = df_targets.attrs.get('original_cols', list(df_targets.columns))
            st.error("⚠️ Sigo sin encontrar la columna 'Date' o 'Target'.")
            st.info(f"🕵️ **Columnas detectadas:** \n\n `{cols_found}`")
    else:
        st.info("⏳ Cargando datos de Targets...")

# ==============================================================================
# 📁 PESTAÑA 3: ANÁLISIS DE CSV LOCAL (COMPLETAMENTE INDEPENDIENTE)
# ==============================================================================
with tab_csv:
    st.subheader("📁 Analiza tu propio CSV (Pestaña 3)")
    st.write("Sube tu archivo `.csv`. ¡Una vez lo subas, aparecerán mágicamente todos los filtros y gráficas idénticos a la pestaña 2!")
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # 1. Leer CSV
            df_raw_csv = pd.read_csv(uploaded_file, sep=None, engine='python')
            df_raw_csv.columns = df_raw_csv.columns.str.strip().str.replace('"', '')
            
            df_debug = df_raw_csv.copy()
            
            # 2. Limpiarlo con la función
            df_cleaned_csv = clean_target_df(df_raw_csv)
            
            # 3. Dibujar UI si todo está OK
            if df_cleaned_csv is not None and not df_cleaned_csv.empty:
                if 'Date' in df_cleaned_csv.columns and 'Target' in df_cleaned_csv.columns:
                    st.success("✅ Archivo cargado y procesado correctamente. ¡Aquí tienes tus datos!")
                    # Filtros con llave "csv"
                    render_targets_ui(df_cleaned_csv, "csv")
                else:
                    cols_found = df_cleaned_csv.attrs.get('original_cols', list(df_cleaned_csv.columns))
                    st.error("⚠️ El CSV subido no tiene las columnas necesarias ('Date' o 'Target').")
                    st.info(f"🕵️ **Columnas que he encontrado:** `{cols_found}`")
            else:
                st.warning("⚠️ El archivo subido se ha quedado vacío tras la limpieza. Es posible que el formato de fecha no se reconozca.")
                st.dataframe(df_debug.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error al leer el archivo CSV: {e}")
    else:
        st.info("👆 Esperando a que subas un archivo .csv para mostrarte las opciones...")
