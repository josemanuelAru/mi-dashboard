import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard Adport/Msales", page_icon="📈", layout="wide")

# --- TU ENLACE ORIGINAL ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/1WuBv1esTxZAfC07BPwWzjsz5TZqfUHa6MOzNIAEOMew/edit?gid=368085162#gid=368085162"

# --- FUNCIÓN PARA CONVERTIR ENLACE DE EDITAR A CSV ---
def get_csv_url(url):
    # Extraemos el ID de la hoja
    sheet_id = url.split('/d/')[1].split('/')[0]
    # Extraemos el GID (el ID de la pestaña específica)
    gid = '0'
    if 'gid=' in url:
        gid = url.split('gid=')[1].split('#')[0]
    
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# --- FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data(ttl=600) # Se actualiza cada 10 minutos
def load_data(original_url):
    csv_url = get_csv_url(original_url)
    try:
        df = pd.read_csv(csv_url)
        
        # Limpiar nombres de columnas (quitar espacios y comillas)
        df.columns = df.columns.str.strip().str.replace('"', '')

        # --- DETECCIÓN INTELIGENTE DE COLUMNAS ---
        # Buscamos la columna de fecha
        date_col = next((col for col in df.columns if col.lower() in ['date', 'day', 'v', 'fecha']), None)
        # Buscamos la columna de coste
        cost_col = next((col for col in df.columns if col.lower() in ['cost', 'coste', 'spend']), None)
        # Buscamos la columna de impresiones/visitas
        imp_col = next((col for col in df.columns if col.lower() in ['impressions', 'impresiones', 'visits', 'received visits zp']), None)
        # Buscamos la columna de país
        country_col = next((col for col in df.columns if col.lower() in ['country', 'pais', 'geo', 'geo/os']), None)
        
        # Procesar FECHA
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(by=date_col)

        # Procesar NÚMEROS (Quitar $ y ,)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Intenta limpiar si parece dinero
                    if df[col].astype(str).str.contains('\$').any():
                        df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df, date_col, cost_col, imp_col, country_col

    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return None, None, None, None, None

# --- CARGAMOS LOS DATOS ---
df, date_col, cost_col, imp_col, country_col = load_data(SHEET_URL)

# --- INTERFAZ DE USUARIO ---
st.title("📊 Dashboard de Control Diario")

if df is not None:
    # FILTROS (SIDEBAR)
    st.sidebar.header("Filtros")
    
    # Filtro de País
    if country_col:
        selected_countries = st.sidebar.multiselect(
            "Filtrar por País:",
            options=df[country_col].unique(),
            default=df[country_col].unique()
        )
        df_filtered = df[df[country_col].isin(selected_countries)]
    else:
        df_filtered = df

    # KPIS PRINCIPALES
    total_cost = df_filtered[cost_col].sum() if cost_col else 0
    total_imps = df_filtered[imp_col].sum() if imp_col else 0
    
    # Mostramos KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("💰 Gasto Total", f"${total_cost:,.2f}")
    kpi2.metric("👁️ Tráfico Total", f"{total_imps:,.0f}")
    
    # Intentamos calcular CPM si tenemos los datos
    if total_imps > 0 and total_cost > 0:
        cpm = (total_cost / total_imps) * 1000
        kpi3.metric("📉 CPM Promedio", f"${cpm:.2f}")

    st.markdown("---")

    # GRÁFICOS
    col1, col2 = st.columns(2)

    # Gráfico 1: Evolución Temporal
    with col1:
        if date_col and cost_col:
            st.subheader("📈 Evolución del Gasto Diario")
            fig_line = px.line(df_filtered, x=date_col, y=cost_col, title="Gasto por Día")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("No se detectaron columnas de Fecha o Coste para el gráfico temporal.")

    # Gráfico 2: Desglose por País
    with col2:
        if country_col and cost_col:
            st.subheader("🌍 Gasto por País")
            fig_bar = px.bar(df_filtered, x=country_col, y=cost_col, color=country_col, title="Inversión por Geo")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No se detectaron columnas de País o Coste para el gráfico.")

    # TABLA DE DATOS
    with st.expander("Ver Datos Brutos"):
        st.dataframe(df_filtered)
        
else:
    st.error("No se pudo conectar con el Excel. Asegúrate de que está compartido como 'Cualquiera con el enlace'.")
