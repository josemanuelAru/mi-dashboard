import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard Master: Cost vs Revenue", page_icon="💰", layout="wide")

# ==========================================
# ⚙️ CONFIGURACIÓN DE DATOS
# ==========================================
SHEET_ID = "1WuBv1esTxZAfC07BPwWzjsz5TZqfUHa6MOzNIAEOMew"
GID_ANDROID = "368085162"    # Datos por AOS/dia
GID_IOS = "1225911759"       # Datos por IOS/dia

def get_url(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

# --- FUNCIÓN DE BÚSQUEDA EXACTA (CORREGIDA) ---
def find_column_strict(columns, candidates):
    """Busca coincidencias exactas primero para evitar errores como 'v' en 'Revenue'."""
    cols_clean = [str(c).lower().strip() for c in columns]
    
    for cand in candidates:
        cand = cand.lower().strip()
        
        # 1. Búsqueda EXACTA (Prioridad absoluta)
        if cand in cols_clean:
            return columns[cols_clean.index(cand)]
            
        # 2. Búsqueda Parcial (Solo si la palabra es larga > 3 letras)
        # Esto evita que 'v' coincida con 'Revenue' o 'Visits'
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
            
            # 2. MAPEO SEGURO
            col_mapping = {}
            
            # Buscamos columnas críticas con lista de sinónimos
            # Nota: 'v' ahora solo coincidirá si la columna se llama exactamente 'v'
            c_date = find_column_strict(df.columns, ['date', 'day', 'fecha', 'v', 'time'])
            c_country = find_column_strict(df.columns, ['country', 'geo', 'geo/os', 'pais', 'region'])
            c_cost = find_column_strict(df.columns, ['cost', 'coste', 'spend', 'total cost'])
            c_rev = find_column_strict(df.columns, ['revenue total', 'revenue', 'ingresos', 'gain'])

            # Métricas ZP
            c_rec_vis = find_column_strict(df.columns, ['received visits zp', 'received visits'])
            c_sold_vis = find_column_strict(df.columns, ['sold visits zp', 'sold visits'])
            c_perc_sold = find_column_strict(df.columns, ['%sold zp', '% sold zp'])
            c_cpm = find_column_strict(df.columns, ['cpm zp', 'cpm'])

            # Asignamos nombres estándar
            if c_date: col_mapping[c_date] = 'Date'
            if c_country: col_mapping[c_country] = 'Country'
            if c_cost: col_mapping[c_cost] = 'Cost'
            if c_rev: col_mapping[c_rev] = 'Revenue'
            if c_rec_vis: col_mapping[c_rec_vis] = 'Received Visits ZP'
            if c_sold_vis: col_mapping[c_sold_vis] = 'Sold Visits ZP'
            if c_perc_sold: col_mapping[c_perc_sold] = '% Sold ZP'
            if c_cpm: col_mapping[c_cpm] = 'CPM ZP'

            # Aplicar cambios
            if col_mapping:
                df.rename(columns=col_mapping, inplace=True)
            
            # 3. VALIDACIÓN DE FECHA
            if 'Date' in df.columns:
                # Convertimos a datetime forzando errores a NaT (Not a Time)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # ¡IMPORTANTE! Eliminamos filas donde la fecha no sea válida (esto arregla tu error)
                df = df.dropna(subset=['Date'])
            else:
                # Si no hay fecha, esta pestaña no sirve
                st.warning(f"⚠️ Saltando {os_name}: No se encontró columna de Fecha.")
                continue

            # 4. LIMPIEZA NUMÉRICA
            numeric_cols = ['Cost', 'Revenue', 'Received Visits ZP', 'Sold Visits ZP', '% Sold ZP', 'CPM ZP']
            for col in numeric_cols:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace('$', '', regex=False)\
                                                     .str.replace(',', '', regex=False)\
                                                     .str.replace('%', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['OS'] = os_name
            
            # Seleccionamos solo las columnas que existen para evitar KeyErrors
            cols_to_keep = ['Date', 'Country', 'OS'] + [c for c in numeric_cols if c in df.columns]
            dfs.append(df[cols_to_keep])
            
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
    
    # --- SIDEBAR: FILTROS ---
    st.sidebar.header("Filtros Globales")
    
    # Rango de fechas seguro
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    if pd.isnull(min_date) or pd.isnull(max_date):
        st.error("⚠️ Error: Las fechas no son válidas. Revisa el formato en el Excel.")
        st.stop()
        
    date_range = st.sidebar.date_input("Rango de Fechas", [min_date, max_date])
    
    selected_os = st.sidebar.multiselect("Sistema Operativo", df['OS'].unique(), default=df['OS'].unique())
    
    if 'Country' in df.columns:
        all_countries = sorted(df['Country'].unique().astype(str))
        selected_countries = st.sidebar.multiselect("Países", all_countries, default=all_countries)
    else:
        selected_countries = []

    # --- APLICAR FILTROS ---
    mask = (df['OS'].isin(selected_os))
    
    if len(date_range) == 2:
        mask = mask & (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
    
    if selected_countries:
        mask = mask & (df['Country'].isin(selected_countries))
        
    df_filtered = df[mask]

    # --- SECCIÓN 1: KPIS ---
    st.subheader("💰 Resumen Financiero")
    
    # Comprobamos existencia de columnas antes de sumar
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

    # --- SECCIÓN 2: GRÁFICA ZP ---
    st.subheader("📈 Análisis de Métricas ZP")
    
    zp_cols = ['Received Visits ZP', 'Sold Visits ZP', '% Sold ZP', 'CPM ZP']
    available_zp = [c for c in zp_cols if c in df_filtered.columns]
    
    col_sel, col_chart = st.columns([1, 3])
    with col_sel:
        st.markdown("**Configuración:**")
        selected_metrics = st.multiselect("Métricas:", options=available_zp, default=available_zp[:2] if available_zp else None)
    
    with col_chart:
        if selected_metrics:
            # Agrupar por fecha
            agg_dict = {m: 'sum' if 'Visits' in m else 'mean' for m in selected_metrics}
            df_zp = df_filtered.groupby('Date')[selected_metrics].agg(agg_dict).reset_index()
            
            fig_zp = px.line(df_zp.melt(id_vars='Date', var_name='Metric', value_name='Value'), 
                             x='Date', y='Value', color='Metric', markers=True)
            st.plotly_chart(fig_zp, use_container_width=True)
        else:
            st.info("Selecciona métricas en el menú izquierdo.")

    # --- SECCIÓN 3: GRÁFICAS FINANCIERAS ---
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📅 Coste vs Revenue")
        if 'Cost' in df_filtered.columns and 'Revenue' in df_filtered.columns:
            df_fin = df_filtered.groupby('Date')[['Cost', 'Revenue']].sum().reset_index()
            fig_fin = px.line(df_fin.melt(id_vars='Date'), x='Date', y='value', color='variable',
                              color_discrete_map={'Cost':'#EF553B', 'Revenue':'#00CC96'})
            st.plotly_chart(fig_fin, use_container_width=True)
            
    with c2:
        st.subheader("🌍 Top Países (Gasto)")
        if 'Country' in df_filtered.columns and 'Cost' in df_filtered.columns:
            top_c = df_filtered.groupby('Country')['Cost'].sum().nlargest(10).reset_index()
            fig_bar = px.bar(top_c, x='Country', y='Cost', color='Cost')
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("⏳ Esperando datos... Si ves esto, revisa que los GIDs sean correctos y el archivo público.")
