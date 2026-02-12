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

# --- FUNCIÓN DE LIMPIEZA INTELIGENTE ---
def find_column(columns, candidates):
    """Busca la primera coincidencia de una lista de candidatos."""
    cols_lower = [str(c).lower().strip() for c in columns]
    for candidate in candidates:
        candidate = candidate.lower().strip()
        # Buscamos coincidencia exacta o parcial segura
        for i, col in enumerate(cols_lower):
            if candidate in col:
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
            
            # 2. MAPEO DE COLUMNAS BÁSICAS
            col_mapping = {}
            
            # Columnas Financieras y Generales
            c_date = find_column(df.columns, ['date', 'day', 'fecha', 'v'])
            c_country = find_column(df.columns, ['country', 'geo', 'geo/os', 'pais'])
            c_cost = find_column(df.columns, ['cost', 'coste', 'spend'])
            c_rev = find_column(df.columns, ['revenue total', 'revenue', 'ingresos', 'gain'])
            
            if c_date: col_mapping[c_date] = 'Date'
            if c_country: col_mapping[c_country] = 'Country'
            if c_cost: col_mapping[c_cost] = 'Cost'
            if c_rev: col_mapping[c_rev] = 'Revenue'

            # 3. MAPEO DE COLUMNAS ZP (Tus nuevas métricas)
            c_rec_visits = find_column(df.columns, ['received visits zp', 'received visits'])
            c_sold_visits = find_column(df.columns, ['sold visits zp', 'sold visits'])
            c_perc_sold = find_column(df.columns, ['%sold zp', '% sold zp', 'sold %'])
            c_cpm_zp = find_column(df.columns, ['cpm zp', 'cpm'])

            if c_rec_visits: col_mapping[c_rec_visits] = 'Received Visits ZP'
            if c_sold_visits: col_mapping[c_sold_visits] = 'Sold Visits ZP'
            if c_perc_sold: col_mapping[c_perc_sold] = '% Sold ZP'
            if c_cpm_zp: col_mapping[c_cpm_zp] = 'CPM ZP'

            # Aplicar renombrado
            df.rename(columns=col_mapping, inplace=True)
            df = df.loc[:, ~df.columns.duplicated()]

            # 4. LIMPIEZA DE DATOS NUMÉRICOS
            # Lista de todas las columnas numéricas posibles
            numeric_cols = ['Cost', 'Revenue', 'Received Visits ZP', 'Sold Visits ZP', '% Sold ZP', 'CPM ZP']
            
            for col in numeric_cols:
                if col in df.columns:
                    # Convertir a string, quitar símbolos raros y pasar a número
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace('$', '', regex=False)\
                                                     .str.replace(',', '', regex=False)\
                                                     .str.replace('%', '', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # 5. Convertir Fecha
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            df['OS'] = os_name
            dfs.append(df)
            
        except Exception as e:
            st.error(f"Error procesando {os_name}: {e}")
    
    if dfs:
        # Concatenar y rellenar con 0 las columnas que falten en alguno de los dos excels
        return pd.concat(dfs, ignore_index=True).fillna(0)
    return None

# --- CARGAR ---
df = load_data()

# --- INTERFAZ ---
st.title("📊 Dashboard Financiero & Métricas ZP")

if df is not None and not df.empty:
    # FILTROS
    st.sidebar.header("Filtros Globales")
    
    min_date, max_date = df['Date'].min(), df['Date'].max()
    date_range = st.sidebar.date_input("Fechas", [min_date, max_date]) if not pd.isnull(min_date) else [None, None]
    
    selected_os = st.sidebar.multiselect("Sistema Operativo", df['OS'].unique(), default=df['OS'].unique())
    all_countries = sorted(df['Country'].unique().astype(str))
    selected_countries = st.sidebar.multiselect("Países", all_countries, default=all_countries)
    
    # FILTRADO
    if len(date_range) == 2 and date_range[0]:
        mask = (
            (df['Date'] >= pd.to_datetime(date_range[0])) & 
            (df['Date'] <= pd.to_datetime(date_range[1])) &
            (df['OS'].isin(selected_os)) &
            (df['Country'].isin(selected_countries))
        )
        df_filtered = df[mask]
    else:
        df_filtered = df

    # --- SECCIÓN 1: FINANCIERA (Resumen) ---
    st.subheader("💰 Resumen Financiero")
    cost = df_filtered['Cost'].sum()
    rev = df_filtered['Revenue'].sum()
    profit = rev - cost
    roi = (profit / cost * 100) if cost > 0 else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Coste Total", f"${cost:,.2f}")
    k2.metric("Revenue Total", f"${rev:,.2f}")
    k3.metric("Beneficio", f"${profit:,.2f}")
    k4.metric("ROI Global", f"{roi:.2f}%")
    
    st.divider()

    # --- SECCIÓN 2: GRÁFICA PERSONALIZADA ZP (NUEVO) ---
    st.subheader("📈 Análisis de Métricas ZP")
    
    # Contenedor para la configuración de la gráfica
    with st.container():
        col_selector, col_graph = st.columns([1, 3])
        
        with col_selector:
            st.markdown("### Configura tu Gráfica")
            st.info("Selecciona qué métricas quieres visualizar:")
            
            # Opciones disponibles
            zp_options = ['Received Visits ZP', 'Sold Visits ZP', '% Sold ZP', 'CPM ZP']
            # Filtramos solo las que existen en el Excel
            available_options = [opt for opt in zp_options if opt in df_filtered.columns]
            
            selected_metrics = st.multiselect(
                "Métricas a mostrar:",
                options=available_options,
                default=['Received Visits ZP', 'Sold Visits ZP'] # Por defecto mostramos visitas
            )
            
            st.warning("Nota: Si mezclas '%' o 'CPM' con 'Visitas', las escalas pueden verse raras. Mejor visualízalas por separado.")

        with col_graph:
            if selected_metrics:
                # Agrupamos por día y sumamos (o promediamos según la métrica)
                # OJO: Para % y CPM no se puede sumar a lo bruto. Hacemos promedio ponderado o simple.
                # Para simplificar la visualización diaria global, usaremos media para tasas y suma para volumen.
                
                df_zp_daily = df_filtered.groupby('Date')[selected_metrics].agg({
                    'Received Visits ZP': 'sum',
                    'Sold Visits ZP': 'sum',
                    '% Sold ZP': 'mean', # Promedio del porcentaje diario
                    'CPM ZP': 'mean'     # Promedio del CPM diario
                }, errors='ignore').reset_index()
                
                # Transformamos para Plotly (Melting)
                df_melted = df_zp_daily.melt(id_vars='Date', value_vars=selected_metrics, var_name='Métrica', value_name='Valor')
                
                fig_zp = px.line(
                    df_melted, 
                    x='Date', 
                    y='Valor', 
                    color='Métrica',
                    title="Evolución de Métricas ZP",
                    markers=True,
                    height=500
                )
                st.plotly_chart(fig_zp, use_container_width=True)
            else:
                st.write("👈 Selecciona al menos una métrica en el menú de la izquierda.")

    st.divider()

    # --- SECCIÓN 3: RESTO DE GRÁFICAS ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Coste vs Revenue Diario")
        daily = df_filtered.groupby('Date')[['Cost', 'Revenue']].sum().reset_index().melt(id_vars='Date')
        st.plotly_chart(px.line(daily, x='Date', y='value', color='variable', 
                        color_discrete_map={'Cost':'#EF553B', 'Revenue':'#00CC96'}), use_container_width=True)
    
    with c2:
        st.subheader("🌍 Top Países (Coste)")
        top_countries = df_filtered.groupby('Country')['Cost'].sum().nlargest(10).reset_index()
        st.plotly_chart(px.bar(top_countries, x='Country', y='Cost', color='Cost'), use_container_width=True)

    with st.expander("Ver Datos Brutos"):
        st.dataframe(df_filtered)

else:
    st.info("Cargando datos... Si esto tarda, revisa los permisos o el formato del Excel.")
